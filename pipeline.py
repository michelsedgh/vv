"""
RealtimePipeline — orchestrates DeepFilterNet → PixIT → WeSpeaker → EnrolledClustering.

Identity follows diart's official embedding path: mixed waveform plus
per-speaker segmentation weights. PixIT-separated channels are used only for
final audio reconstruction.

Data flow per step
──────────────────
  waveform np.ndarray (80000,) — 5 s at 16 kHz
    → [optional] DeepFilterNet denoise (16k→48k→denoise→16k)
    → PixIT                → diarization SlidingWindowFeature (624, 3)
                           → sources tensor (1, 80000, 3)
    → WeSpeaker on mixed waveform + diarization weights → (3, emb_dim) embeddings
    → EnrolledSpeakerClustering → permuted_seg + SpeakerMap (local → global)
    → DIART DelayedAggregation (Hamming) on permuted_seg → smoother activity
      (same idea as diart SpeakerDiarization.pred_aggregation; clustering still
      runs on the instantaneous permuted chunk like DIART)
    → source mapping: per-speaker step audio + activity from aggregated seg
    → [optional] DeepFilterNet on separated audio for enrolled speakers only

Playback quality and output alignment
────────────────────────────────────
PixIT is re-run every 500 ms on a sliding 5 s window, so the newest 500 ms
slice exists in only one separator window. Emitting that freshest slice
directly makes onsets weak and unstable. The backend therefore emits a
**delayed step** whose samples are already covered by at least two overlapping
PixIT windows. This mirrors diart's latency-oriented design: real time with a
controlled output delay rather than zero-latency edge audio.

Clustered separated sources are also leakage-pruned before output. PixIT always
produces a fixed number of local sources, and more than one local can contain
the same real speaker. Those secondary locals must be treated as leakage of the
matched enrolled speaker, not as new ``Unknown-*`` speakers.

See also: module docstring in ``enrolled_clustering.py`` and comments in
``config.yaml`` for tunables.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio.functional as AF
import yaml
from pyannote.audio import Model as PyanModel
from pyannote.core import SlidingWindow, SlidingWindowFeature

from diart.blocks.aggregation import DelayedAggregation
from diart.blocks.embedding import OverlapAwareSpeakerEmbedding
from enrolled_clustering import EnrolledSpeakerClustering
from pixit_wrapper import make_pixit_segmentation_model

log = logging.getLogger(__name__)


# ─── Output dataclass ────────────────────────────────────────────

@dataclass
class SpeakerResult:
    """Per-speaker output for a single pipeline step."""
    global_idx: int
    label: str
    audio: np.ndarray          # float32, 16 kHz, mono — last step_samples
    activity: float            # mean local activity over the emitted 500 ms slice
    is_enrolled: bool
    # WeSpeaker: cosine similarity (dot product) of this step's overlap-aware
    # embedding vs frozen enrollment anchor — only for enrolled; None otherwise.
    # Orthogonal to ``activity`` (which is separator activity strength).
    identity_similarity: Optional[float] = None


@dataclass
class StepResult:
    """Full output of one pipeline step."""
    speakers: List[SpeakerResult] = field(default_factory=list)
    infer_ms: float = 0.0
    step_idx: int = 0


# ─── Pipeline ────────────────────────────────────────────────────

class RealtimePipeline:
    """Real-time speaker diarization + separation pipeline.

    End-to-end responsibilities:

    - Run PixIT on each 5 s chunk; read diarization scores and separated tracks.
    - Extract overlap-aware WeSpeaker embeddings from the mixed chunk.
    - Feed ``EnrolledSpeakerClustering`` for local→global IDs.
    - Reconstruct global separated sources across overlapping chunks.
    - Emit a delayed 500 ms slice. Enrolled live playback uses delayed mixed
      audio gated by the assigned speaker activity mask, because raw separated
      PixIT locals are not stable causal speaker streams at onsets.

    Parameters
    ----------
    config : dict
        Parsed config.yaml contents.
    enrolled_embeddings : dict[str, np.ndarray]
        Speaker name → 256-dim WeSpeaker embedding (L2-normed).
    """

    def __init__(
        self,
        config: dict,
        enrolled_embeddings: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.cfg = config
        _dbg = config.get("debug", {})
        self._step_perf_log = bool(_dbg.get("step_perf_log", False))
        self._step_perf_every_step = bool(_dbg.get("step_perf_every_step", False))
        self._step_perf_spike_threshold_ms = float(
            _dbg.get("step_perf_spike_threshold_ms", 300.0)
        )
        self._step_perf_path: Optional[Path] = None
        self._step_perf_session_id: Optional[str] = None
        if self._step_perf_log:
            perf_path = Path(
                _dbg.get("step_perf_ndjson", "logs/infer_step_perf.ndjson")
            )
            if not perf_path.is_absolute():
                perf_path = Path(__file__).parent / perf_path
            perf_path.parent.mkdir(parents=True, exist_ok=True)
            if bool(_dbg.get("step_perf_reset_on_init", True)):
                perf_path.write_text("", encoding="utf-8")
            self._step_perf_path = perf_path
            self._step_perf_session_id = time.strftime("%Y%m%d-%H%M%S")
        self.device = torch.device(config["device"])
        self.sample_rate: int = config["audio"]["sample_rate"]

        pixit_cfg = config["pixit"]
        self.duration: float = pixit_cfg["duration"]
        self.step_duration: float = pixit_cfg["step"]
        self.chunk_samples = int(self.duration * self.sample_rate)   # 80000
        self.step_samples = int(self.step_duration * self.sample_rate)  # 8000
        self.n_local_speakers: int = pixit_cfg["max_speakers"]       # 3

        # DIART SpeakerDiarization.pred_aggregation: Hamming-weighted fusion of
        # overlapping permuted segmentations.  Reduces frame-to-frame jitter in
        # activity (and stabilizes UI); latency_sec >= step (default 2 s ⇒ 4×0.5 s windows).
        _agg_lat = float(pixit_cfg.get("aggregation_latency_sec", 2.0))
        if _agg_lat < self.step_duration:
            _agg_lat = self.step_duration
        self._pred_aggregation = DelayedAggregation(
            self.step_duration,
            latency=_agg_lat,
            strategy="hamming",
            cropping_mode="loose",
        )
        self._pred_seg_buffer: List[SlidingWindowFeature] = []
        log.info(
            "Diarization aggregation (DIART): step=%.3fs latency=%.3fs → %d windows",
            self.step_duration,
            _agg_lat,
            self._pred_aggregation.num_overlapping_windows,
        )

        emb_cfg = config["embedding"]
        clust_cfg = config["clustering"]

        # ── CUDA optimisation ──
        torch.backends.cudnn.benchmark = True       # auto-tune convs for fixed 80k input

        # ── Segmentation (PixIT) ──
        log.info("Loading PixIT segmentation model...")
        seg_model = make_pixit_segmentation_model(pixit_cfg["model"])
        seg_model.to(self.device)                    # triggers lazy load + device placement
        seg_model.eval()
        self._pixit_wrapper = seg_model.model        # PixITWrapper nn.Module (now loaded)

        # ── Optimise: permanent fp16 (286ms → 250ms) ──
        self._pixit_wrapper.half()
        log.info("Converted PixIT to permanent fp16.")

        # ── Optimise: TRT encoder (80ms → 29ms for WavLM encoder) ──
        from trt_wavlm import patch_wavlm_with_trt
        wavlm = self._pixit_wrapper.totatonet.wavlm
        self._trt_active = patch_wavlm_with_trt(wavlm, self.device)

        log.info("Loading WeSpeaker embedding model...")
        self._emb_model = PyanModel.from_pretrained(emb_cfg["model"]).to(self.device)
        self._emb_model.eval()

        # Unknown-speaker clustering uses diart's official path: one mixed
        # waveform chunk plus per-speaker segmentation weights. Persistent
        # enrolled verification uses fixed-length embeddings from PixIT's
        # separated sources so enrollment/live verification stay in the same
        # domain and avoid frame-mismatch warnings in the weighted path.
        self._embed_seg_threshold = float(
            emb_cfg.get("source_seg_threshold", max(0.12, clust_cfg["tau_active"] * 0.5))
        )
        self._embed_min_voiced_sec = float(emb_cfg.get("source_min_voiced_sec", 0.04))
        self._embed_recent_sec = float(
            emb_cfg.get("source_recent_sec", min(2.0, self.duration))
        )
        self._embed_recent_sec = min(max(self._embed_recent_sec, 0.25), self.duration)
        self._enroll_profile_top_k = max(
            1, int(emb_cfg.get("enrollment_profile_top_k", 3))
        )
        self._embed_gamma = float(emb_cfg.get("gamma", 3.0))
        self._embed_beta = float(emb_cfg.get("beta", 10.0))
        self._oa_emb = OverlapAwareSpeakerEmbedding(
            self._emb_model,
            gamma=self._embed_gamma,
            beta=self._embed_beta,
            device=self.device,
        )
        log.info(
            "Embedding path: overlap-aware mixed-waveform embeddings (recent=%.2fs min_voiced=%.3fs gamma=%.2f beta=%.2f)",
            self._embed_recent_sec,
            self._embed_min_voiced_sec,
            self._embed_gamma,
            self._embed_beta,
        )
        self._onset_aux_max_voiced_sec = float(
            clust_cfg.get("onset_aux_max_voiced_sec", 1.5)
        )
        self._onset_aux_dominance_ratio = float(
            clust_cfg.get("onset_aux_dominance_ratio", 1.25)
        )
        self._onset_aux_dist_margin = float(
            clust_cfg.get("onset_aux_dist_margin", 0.08)
        )

        # ── DeepFilterNet (optional) — load BEFORE enrollment ──
        # Mic path (`enabled`) and enrolled-output path (`enhance_enrolled_output`)
        # are independent: output enhancement does not touch the buffer fed to PixIT.
        denoiser_cfg = config.get("denoiser", {})
        self._denoiser_enabled = denoiser_cfg.get("enabled", False)
        self._df_output_enrolled = denoiser_cfg.get("enhance_enrolled_output", False)
        self._df_model = None
        self._df_state_mic = None
        self._df_enhance = None
        self._df_sr = 48_000
        self._df_output_states: Dict[int, Any] = {}

        _load_df = self._denoiser_enabled or self._df_output_enrolled
        if _load_df:
            try:
                from df import enhance, init_df
                model_name = denoiser_cfg.get("model", "DeepFilterNet3")
                log.info(
                    "Loading DeepFilterNet (%s) [mic=%s, enrolled_output=%s]...",
                    model_name,
                    self._denoiser_enabled,
                    self._df_output_enrolled,
                )
                self._df_model, df_state, _ = init_df(model_base_dir=model_name)
                self._df_enhance = enhance
                self._df_sr = df_state.sr()
                if self._denoiser_enabled:
                    self._df_state_mic = df_state
                log.info("DeepFilterNet ready (native %d Hz).", self._df_sr)
            except Exception as e:
                log.warning("DeepFilterNet failed to load, disabling: %s", e)
                self._denoiser_enabled = False
                self._df_output_enrolled = False
                self._df_model = None
                self._df_state_mic = None
                self._df_enhance = None

        if self._df_model is not None:
            from torchaudio.transforms import Resample
            self._df_resample_up = Resample(self.sample_rate, self._df_sr)
            self._df_resample_down = Resample(self._df_sr, self.sample_rate)

        # ── Clustering ──
        self.clustering = EnrolledSpeakerClustering(
            tau_active=clust_cfg["tau_active"],
            rho_update=clust_cfg["rho_update"],
            delta_new=clust_cfg["delta_new"],
            metric="cosine",
            max_speakers=clust_cfg["max_speakers"],
            max_drift=clust_cfg.get("max_drift", 0.0),
            delta_enrolled=clust_cfg.get("delta_enrolled", 0.80),
            leakage_delta=clust_cfg.get("leakage_delta"),
        )
        self._enrollment_profiles_by_name: Dict[str, np.ndarray] = {}
        if enrolled_embeddings:
            # Re-extract through EXACT same path as live: [DF→]PixIT→WeSpeaker
            reextracted = self._reextract_enrollments(enrolled_embeddings, config)
            self.clustering.inject_centroids(reextracted)
            self.clustering.inject_profiles(self._enrollment_profiles_by_name)
            log.info(
                "Injected %d enrolled speaker(s): %s",
                len(reextracted),
                ", ".join(reextracted.keys()),
            )

        # ── State ──
        self._step_idx = 0
        acfg = config.get("audio", {})
        self._output_latency_sec = float(
            acfg.get("output_latency_sec", _agg_lat)
        )
        self._output_latency_sec = min(
            self.duration,
            max(self.step_duration, self._output_latency_sec),
        )
        self._output_latency_steps = max(
            1, int(round(self._output_latency_sec / self.step_duration))
        )
        self._enrolled_live_mix = bool(acfg.get("enrolled_live_mix", True))
        self._enrolled_mix_gate_threshold = float(
            acfg.get(
                "enrolled_mix_gate_threshold",
                max(0.06, self.cfg["clustering"]["tau_active"] * 0.25),
            )
        )
        self._enrolled_mix_gate_collar_sec = float(
            acfg.get("enrolled_mix_gate_collar_sec", 0.03)
        )
        log.info(
            "Separated-audio output latency: %.3fs (%d step%s)",
            self._output_latency_sec,
            self._output_latency_steps,
            "" if self._output_latency_steps == 1 else "s",
        )
        self._audio_aggregation = DelayedAggregation(
            self.step_duration,
            latency=self._output_latency_sec,
            strategy="first",
            cropping_mode="center",
        )
        # Audio gating must use a prediction track aligned to the SAME delayed
        # step as emitted audio. Using the 2.0 s UI/activity aggregation here
        # would shift the gate by multiple 500 ms steps and create periodic
        # packet-boundary dropouts.
        self._emit_pred_aggregation = DelayedAggregation(
            self.step_duration,
            latency=self._output_latency_sec,
            strategy="hamming",
            cropping_mode="loose",
        )
        self._audio_buffer: List[SlidingWindowFeature] = []
        self._mix_buffer: List[SlidingWindowFeature] = []
        self._emit_pred_buffer: List[SlidingWindowFeature] = []
        self._emit_meta_buffer: List[tuple[int, List[Dict[str, Any]]]] = []
        if self._step_perf_log and self._step_perf_path is not None:
            log.info(
                "Step perf trace enabled: %s (every_step=%s, spike>=%.1f ms)",
                self._step_perf_path,
                self._step_perf_every_step,
                self._step_perf_spike_threshold_ms,
            )

        # Warmup both models + detect embedding dim
        log.info("Warming up PixIT + WeSpeaker...")
        dummy_fp16 = torch.randn(1, 1, self.chunk_samples, device=self.device, dtype=torch.float16)
        dummy_fp32 = dummy_fp16.float()
        with torch.inference_mode():
            self._pixit_wrapper(dummy_fp16)       # PixIT is fp16
            d = self._emb_model(dummy_fp32)       # WeSpeaker stays fp32
            self._emb_dim = d.shape[-1]
            dummy_waveform = torch.randn(
                1, self.chunk_samples, 1, device=self.device, dtype=torch.float32
            )
            dummy_seg = torch.rand(
                1, 624, self.n_local_speakers, device=self.device, dtype=torch.float32
            )
            self._oa_emb(dummy_waveform, dummy_seg)
        torch.cuda.synchronize(self.device)
        log.info("Pipeline ready (emb_dim=%d).", self._emb_dim)

    def _perf_log_append(self, payload: dict) -> None:
        if not self._step_perf_log or self._step_perf_path is None:
            return
        rec = {
            "session_id": self._step_perf_session_id,
            "ts_ms": int(time.time() * 1000),
            **payload,
        }
        with self._step_perf_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")

    # ─── DeepFilterNet streaming denoise ─────────────────────────

    def _df_enhance_block(self, block: np.ndarray, df_state: Any) -> np.ndarray:
        """Run one DeepFilterNet enhancement block (16 kHz → 48 kHz → DF → 16 kHz)."""
        n = len(block)
        wav_t = torch.from_numpy(np.asarray(block, dtype=np.float32)).unsqueeze(0)
        wav_48 = self._df_resample_up(wav_t)
        wav_48 = self._df_enhance(self._df_model, df_state, wav_48)
        wav_16 = self._df_resample_down(wav_48)
        return wav_16.squeeze(0).numpy()[:n]

    def denoise_block(self, block: np.ndarray) -> np.ndarray:
        """Denoise a short audio block (e.g. 500 ms) using DeepFilterNet.

        Call this on each mic block BEFORE adding to the rolling buffer.
        DF state is maintained between calls for streaming continuity.
        Runs entirely on CPU — no GPU transfers needed.

        Parameters
        ----------
        block : np.ndarray, shape (samples,), float32, 16 kHz

        Returns
        -------
        np.ndarray, shape (samples,), float32, 16 kHz, denoised
        """
        if not self._denoiser_enabled or self._df_model is None or self._df_state_mic is None:
            return block
        return self._df_enhance_block(block, self._df_state_mic)

    def _df_output_state(self, global_idx: int) -> Any:
        """Streaming DF state per global speaker (enrolled output path only)."""
        if global_idx not in self._df_output_states:
            from df.model import ModelParams
            from libdf import DF as LibdfDF

            p = ModelParams()
            self._df_output_states[global_idx] = LibdfDF(
                sr=p.sr,
                fft_size=p.fft_size,
                hop_size=p.hop_size,
                nb_bands=p.nb_erb,
                min_nb_erb_freqs=p.min_nb_freqs,
            )
        return self._df_output_states[global_idx]

    def _enhance_enrolled_audio(self, samples: np.ndarray, global_idx: int) -> np.ndarray:
        """DeepFilterNet on separated track; chunked like mic path for stable streaming.

        Note: 500 ms internal blocks can introduce audible level steps at block
        edges when this path is enabled — prefer ``enhance_enrolled_output: false``
        unless denoised playback is required (see module docstring).
        """
        if (
            not self._df_output_enrolled
            or self._df_model is None
            or self._df_enhance is None
        ):
            return samples
        samples = np.asarray(samples, dtype=np.float32).flatten()
        state = self._df_output_state(global_idx)
        block_size = int(0.5 * self.sample_rate)
        n = len(samples)
        if n == 0:
            return samples
        out_parts: List[np.ndarray] = []
        for start in range(0, n, block_size):
            chunk = samples[start : start + block_size]
            take = len(chunk)
            if len(chunk) < block_size:
                chunk = np.pad(chunk, (0, block_size - len(chunk)))
            enhanced = self._df_enhance_block(chunk, state)
            out_parts.append(enhanced[:take])
        merged = np.concatenate(out_parts)
        return np.clip(merged, -1.0, 1.0).astype(np.float32)

    # ─── Embedding helpers ────────────────────────────────────────

    def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract L2-normalized WeSpeaker embedding from raw mono audio.

        Used as the raw enrollment fallback path when PixIT cannot provide a
        speech-valid weighted embedding candidate.

        Parameters
        ----------
        audio : np.ndarray, shape (samples,), float32, 16 kHz

        Returns
        -------
        np.ndarray, shape (emb_dim,), float64, L2-normalized
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0)
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._emb_model(wav)   # (1, emb_dim)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def _extract_source_verify_embeddings(
        self,
        sources: torch.Tensor,
        active_indices: List[int],
    ) -> torch.Tensor:
        """Extract fixed-length direct-source embeddings for enrolled verification."""
        verify = torch.full(
            (self.n_local_speakers, self._emb_dim), float("nan"), device="cpu"
        )
        if not active_indices:
            return verify

        keep_samples = min(
            self.chunk_samples,
            max(1, int(round(self._embed_recent_sec * self.sample_rate))),
        )
        batch: List[np.ndarray] = []
        batch_indices: List[int] = []
        for spk_idx in active_indices:
            src = (
                sources[0, -keep_samples:, spk_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            peak = float(np.max(np.abs(src))) if src.size else 0.0
            if peak <= 1e-6:
                continue
            src = np.clip(src / peak, -1.0, 1.0)
            batch.append(src)
            batch_indices.append(int(spk_idx))

        if not batch:
            return verify

        wav = torch.from_numpy(np.stack(batch, axis=0)).unsqueeze(1).to(
            device=self.device, dtype=torch.float32
        )
        with torch.inference_mode():
            embs = self._emb_model(wav)
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-8)
        embs = embs.cpu()
        for row_idx, spk_idx in enumerate(batch_indices):
            verify[spk_idx] = embs[row_idx]
        return verify

    def _source_stats(
        self,
        source_audio: torch.Tensor,
        seg_track: np.ndarray,
        recent_sec: Optional[float] = None,
    ) -> Dict[str, float]:
        """Summarize one separated source for embedding/clustering decisions."""
        src = source_audio.float().flatten()
        seg_arr = np.asarray(seg_track, dtype=np.float32).reshape(-1)
        if recent_sec is not None and recent_sec < self.duration:
            keep_samples = min(
                int(round(recent_sec * self.sample_rate)),
                int(src.numel()),
            )
            if keep_samples > 0 and keep_samples < int(src.numel()):
                src = src[-keep_samples:]
            if len(seg_arr) > 0:
                keep_frames = max(
                    1,
                    int(round(len(seg_arr) * min(1.0, keep_samples / float(max(1, source_audio.numel()))))),
                )
                if keep_frames < len(seg_arr):
                    seg_arr = seg_arr[-keep_frames:]
        peak = float(src.abs().max())
        window_sec = (
            min(self.duration, float(src.numel()) / float(self.sample_rate))
            if src.numel() > 0
            else 0.0
        )
        frame_sec = window_sec / float(max(1, len(seg_arr)))
        voiced_frames = int(np.count_nonzero(seg_arr >= self._embed_seg_threshold))
        stats = {
            "peak": peak,
            "voiced_sec": voiced_frames * frame_sec,
            "voiced_ratio": voiced_frames / float(max(1, len(seg_arr))),
            "seg_max": float(np.max(seg_arr)) if len(seg_arr) else 0.0,
            "seg_mean": float(np.mean(seg_arr)) if len(seg_arr) else 0.0,
        }
        return stats

    def _embedding_window(
        self,
        waveform: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop waveform + segmentation to the recent fixed window used for verification."""
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        segmentation = np.asarray(segmentation, dtype=np.float32)
        if self._embed_recent_sec >= self.duration:
            return waveform, segmentation
        keep_samples = min(
            int(round(self._embed_recent_sec * self.sample_rate)),
            int(waveform.shape[0]),
        )
        if keep_samples <= 0 or keep_samples >= waveform.shape[0]:
            return waveform, segmentation
        keep_frames = max(
            1,
            int(round(segmentation.shape[0] * keep_samples / float(max(1, waveform.shape[0])))),
        )
        return waveform[-keep_samples:], segmentation[-keep_frames:, :]

    def _chunk_start_time(self, step_idx: int) -> float:
        """Absolute start time of the chunk processed at ``step_idx``."""
        return step_idx * self.step_duration - (self.duration - self.step_duration)

    def _aggregate_audio_chunk(self, step_idx: int, chunk_sources: np.ndarray) -> np.ndarray:
        """Aggregate one global-aligned separated chunk using diart's audio policy."""
        audio_sw = SlidingWindow(
            start=self._chunk_start_time(step_idx),
            duration=1.0 / float(self.sample_rate),
            step=1.0 / float(self.sample_rate),
        )
        self._audio_buffer.append(SlidingWindowFeature(chunk_sources, audio_sw))
        agg_audio = self._audio_aggregation(self._audio_buffer)
        if len(self._audio_buffer) == self._audio_aggregation.num_overlapping_windows:
            self._audio_buffer = self._audio_buffer[1:]

        audio = np.asarray(agg_audio.data, dtype=np.float32)
        if audio.shape[0] < self.step_samples:
            audio = np.pad(audio, ((0, self.step_samples - audio.shape[0]), (0, 0)))
        elif audio.shape[0] > self.step_samples:
            audio = audio[: self.step_samples]
        return audio

    def _aggregate_mix_chunk(self, step_idx: int, chunk_waveform: np.ndarray) -> np.ndarray:
        """Aggregate one mono chunk with the same delay policy as speaker audio."""
        audio_sw = SlidingWindow(
            start=self._chunk_start_time(step_idx),
            duration=1.0 / float(self.sample_rate),
            step=1.0 / float(self.sample_rate),
        )
        mono = np.asarray(chunk_waveform, dtype=np.float32).reshape(-1, 1)
        self._mix_buffer.append(SlidingWindowFeature(mono, audio_sw))
        agg_audio = self._audio_aggregation(self._mix_buffer)
        if len(self._mix_buffer) == self._audio_aggregation.num_overlapping_windows:
            self._mix_buffer = self._mix_buffer[1:]

        audio = np.asarray(agg_audio.data, dtype=np.float32).reshape(-1)
        if audio.shape[0] < self.step_samples:
            audio = np.pad(audio, (0, self.step_samples - audio.shape[0]))
        elif audio.shape[0] > self.step_samples:
            audio = audio[: self.step_samples]
        return audio

    def _step_track_mask(self, step_track: np.ndarray, collar_sec: float) -> np.ndarray:
        """Upsample one delayed global activity track to a sample gate mask.

        The important fix is *which* track we gate with: the delayed aggregated
        global speaker track, not the raw current-step local track. We still use
        a binary speech gate here so enrolled playback keeps its level instead
        of collapsing to very low volume on every low-confidence frame.
        """
        track = np.asarray(step_track, dtype=np.float32).reshape(-1)
        if track.size == 0:
            return np.zeros(self.step_samples, dtype=np.float32)
        track = np.nan_to_num(track, nan=0.0, posinf=1.0, neginf=0.0)
        track = np.clip(track, 0.0, 1.0)
        active = track >= float(self._enrolled_mix_gate_threshold)
        if collar_sec > 0.0:
            frame_sec = self.step_duration / float(max(1, track.size))
            collar_frames = max(0, int(round(collar_sec / frame_sec)))
            if collar_frames > 0:
                kernel = np.ones(2 * collar_frames + 1, dtype=np.float32)
                active = np.convolve(active.astype(np.float32), kernel, mode="same") > 0
        frame_pos = np.linspace(0.0, self.step_samples - 1, track.size, dtype=np.float32)
        sample_pos = np.arange(self.step_samples, dtype=np.float32)
        mask = np.interp(sample_pos, frame_pos, active.astype(np.float32))
        return mask.astype(np.float32)

    def _source_sample_mask(self, seg_track: np.ndarray, collar_sec: float = 0.05) -> np.ndarray:
        """Build a sample-rate activity mask from one local segmentation track."""
        seg_arr = np.asarray(seg_track, dtype=np.float32).reshape(-1)
        if seg_arr.size == 0:
            return np.zeros(self.chunk_samples, dtype=np.float32)
        active = seg_arr >= self._embed_seg_threshold
        if collar_sec > 0.0:
            frame_sec = self.duration / float(max(1, seg_arr.size))
            collar_frames = max(0, int(round(collar_sec / frame_sec)))
            if collar_frames > 0:
                kernel = np.ones(2 * collar_frames + 1, dtype=np.float32)
                active = np.convolve(active.astype(np.float32), kernel, mode="same") > 0
        frame_pos = np.linspace(0.0, self.chunk_samples - 1, seg_arr.size, dtype=np.float32)
        sample_pos = np.arange(self.chunk_samples, dtype=np.float32)
        mask = np.interp(sample_pos, frame_pos, active.astype(np.float32))
        return mask.astype(np.float32)

    def _onset_aux_enrolled_pair(
        self,
        pairs: List[tuple[int, int]],
        active_indices: List[int],
        source_stats: List[Dict[str, float]],
        step_tail_rms_np: np.ndarray,
        embeddings: torch.Tensor,
    ) -> tuple[List[tuple[int, int]], int]:
        """Resolve short enrolled onsets when PixIT splits speech across locals.

        On first-word onsets, the enrolled anchor can briefly match a weaker
        leakage local while the dominant separated local is only slightly
        farther from the anchor. That produces the exact symptom the user hears:
        correct identity, but a low-volume first syllable. Prefer the dominant
        local when it is still close enough to the enrolled anchor.
        """
        if len(self.clustering._anchors) != 1 or not active_indices:
            return pairs, 0

        ranked = sorted(
            (int(idx) for idx in active_indices),
            key=lambda idx: float(step_tail_rms_np[idx]),
            reverse=True,
        )
        local_idx = ranked[0]
        if len(ranked) > 1:
            second = float(step_tail_rms_np[ranked[1]])
            if float(step_tail_rms_np[local_idx]) < (second * self._onset_aux_dominance_ratio):
                return pairs, 0
        if source_stats[local_idx]["voiced_sec"] > self._onset_aux_max_voiced_sec:
            return pairs, 0

        anchor_idx = next(iter(self.clustering._anchors))
        dominant_dist = float(
            self.clustering.enrolled_distance(
                anchor_idx,
                embeddings[local_idx].detach().cpu().numpy().astype(np.float64, copy=False),
                local_idx,
            )
        )

        enrolled_pairs = [
            (int(local), int(global_idx))
            for local, global_idx in pairs
            if self.clustering.is_enrolled(int(global_idx))
        ]
        if not enrolled_pairs:
            if dominant_dist > self.clustering.delta_enrolled:
                return pairs, 1
            out_pairs = [(l, g) for l, g in pairs if l != local_idx]
            out_pairs.append((local_idx, anchor_idx))
            return out_pairs, 1

        assigned_local, _ = enrolled_pairs[0]
        if assigned_local == local_idx:
            return pairs, 1

        assigned_dist = float(
            self.clustering.enrolled_distance(
                anchor_idx,
                embeddings[assigned_local].detach().cpu().numpy().astype(np.float64, copy=False),
                assigned_local,
            )
        )
        tail_gain = float(step_tail_rms_np[local_idx]) / max(
            float(step_tail_rms_np[assigned_local]),
            1e-6,
        )
        dist_margin = float(getattr(self, "_onset_aux_dist_margin", 0.08))
        should_swap = (
            tail_gain >= self._onset_aux_dominance_ratio
            and dominant_dist <= (self.clustering.delta_enrolled + dist_margin)
            and dominant_dist <= (assigned_dist + dist_margin)
        )
        if not should_swap:
            return pairs, 1

        out_pairs = [
            (l, g)
            for l, g in pairs
            if int(g) != anchor_idx and int(l) != local_idx
        ]
        out_pairs.append((local_idx, anchor_idx))
        return out_pairs, 1

    def _reextract_enrollments(
        self,
        enrolled_embeddings: Dict[str, np.ndarray],
        config: dict,
    ) -> Dict[str, np.ndarray]:
        """Re-extract enrollment embeddings through [DF→]PixIT→WeSpeaker.

        Runs reference.wav through the EXACT same path as live audio:
        DeepFilterNet (if enabled) → PixIT separation → WeSpeaker embedding.
        This eliminates any domain gap between enrollment and live embeddings.
        """
        import soundfile as sf
        enroll_dir = Path(__file__).parent / config["enrollment"]["dir"]
        reextracted = {}
        self._enrollment_profiles_by_name = {}
        for name in enrolled_embeddings:
            wav_path = enroll_dir / name / "reference.wav"
            if not wav_path.exists():
                log.warning("No reference.wav for '%s', using stored embedding", name)
                reextracted[name] = enrolled_embeddings[name]
                self._enrollment_profiles_by_name[name] = np.asarray(
                    enrolled_embeddings[name], dtype=np.float64
                ).reshape(1, -1)
                continue

            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != self.sample_rate:
                audio_t = torch.from_numpy(audio).unsqueeze(0)
                audio_t = AF.resample(audio_t, sr, self.sample_rate)
                audio = audio_t.squeeze(0).numpy()

            # DeepFilterNet: process in 500ms blocks (same as live path)
            if self._denoiser_enabled and self._df_model is not None:
                block_size = int(0.5 * self.sample_rate)  # 8000 samples
                denoised = []
                for start in range(0, len(audio), block_size):
                    block = audio[start:start + block_size]
                    if len(block) < block_size:
                        block = np.pad(block, (0, block_size - len(block)))
                    block = self.denoise_block(block)
                    denoised.append(block)
                audio = np.concatenate(denoised)[:len(audio)]
                log.info("  '%s' passed through DeepFilterNet (same as live)", name)

            # PixIT is a 5 s model, so search the whole reference file for the
            # best speech-bearing separated source instead of truncating to the
            # first 5 seconds.
            if len(audio) <= self.chunk_samples:
                chunk_starts = [0]
            else:
                chunk_starts = list(range(0, len(audio) - self.chunk_samples + 1, self.chunk_samples))
                tail_start = len(audio) - self.chunk_samples
                if chunk_starts[-1] != tail_start:
                    chunk_starts.append(tail_start)

            target_emb = np.asarray(enrolled_embeddings[name], dtype=np.float64)
            raw_emb = self.extract_embedding(audio)
            best_emb: Optional[np.ndarray] = None
            best_stats: Optional[Dict[str, float]] = None
            best_spk: Optional[int] = None
            best_start: Optional[int] = None
            best_dist: Optional[float] = None
            candidate_embs: List[np.ndarray] = []
            candidate_scores: List[tuple[float, float, float, float, float]] = []

            with torch.inference_mode():
                for start in chunk_starts:
                    chunk = audio[start : start + self.chunk_samples]
                    if len(chunk) < self.chunk_samples:
                        chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

                    wav_gpu = torch.from_numpy(chunk).to(device=self.device, dtype=torch.float16)
                    wav_gpu = wav_gpu.unsqueeze(0).unsqueeze(0)
                    seg_t = self._pixit_wrapper(wav_gpu)
                    seg_np = seg_t[0].float().cpu().numpy()
                    sources = self._pixit_wrapper.last_sources.float()

                    batch_meta: List[tuple[int, Dict[str, float]]] = []
                    for spk_idx in range(seg_np.shape[1]):
                        stats = self._source_stats(
                            sources[0, :, spk_idx],
                            seg_np[:, spk_idx],
                            recent_sec=self._embed_recent_sec,
                        )
                        if (
                            stats["voiced_sec"] < self._embed_min_voiced_sec
                        ):
                            continue
                        batch_meta.append((spk_idx, stats))

                    if not batch_meta:
                        continue

                    for spk_idx, stats in batch_meta:
                        src = sources[0, :, spk_idx].detach().cpu().numpy().astype(np.float32)
                        keep_samples = min(
                            len(src),
                            max(1, int(round(self._embed_recent_sec * self.sample_rate))),
                        )
                        if keep_samples <= 0:
                            continue
                        emb = self.extract_embedding(src[-keep_samples:])
                        from scipy.spatial.distance import cosine as cos_dist
                        dist = float(cos_dist(emb, target_emb))
                        score = (
                            dist,
                            -stats["voiced_sec"],
                            -stats["seg_mean"],
                            -stats["seg_max"],
                            -stats["peak"],
                        )
                        best_score = (
                            best_dist,
                            -best_stats["voiced_sec"],
                            -best_stats["seg_mean"],
                            -best_stats["seg_max"],
                            -best_stats["peak"],
                        ) if best_stats is not None and best_dist is not None else None
                        if best_score is None or score < best_score:
                            best_emb = emb
                            best_stats = stats
                            best_spk = spk_idx
                            best_start = start
                            best_dist = dist
                        candidate_embs.append(emb)
                        candidate_scores.append(score)

            if candidate_embs:
                order = sorted(range(len(candidate_embs)), key=lambda i: candidate_scores[i])
                keep = order[: min(self._enroll_profile_top_k, len(order))]
                profile_rows = np.stack([candidate_embs[i] for i in keep], axis=0)
                profile_rows = np.concatenate(
                    [profile_rows, raw_emb.reshape(1, -1)],
                    axis=0,
                )
                prof = profile_rows.mean(axis=0)
                prof_norm = np.linalg.norm(prof)
                if prof_norm > 0:
                    prof = prof / prof_norm
                emb = prof.astype(np.float64)
                self._enrollment_profiles_by_name[name] = profile_rows.astype(
                    np.float64, copy=False
                )
            else:
                emb = raw_emb
                self._enrollment_profiles_by_name[name] = raw_emb.reshape(1, -1)

            from scipy.spatial.distance import cosine as cos_dist
            raw_d = cos_dist(raw_emb, enrolled_embeddings[name])
            sep_d = cos_dist(emb, enrolled_embeddings[name])
            if best_stats is None:
                log.warning(
                    "  '%s' had no speech-valid PixIT source; falling back to raw enrollment embedding",
                    name,
                )
            else:
                log.info(
                    "  '%s' PixIT chunk=%0.2fs local=%d voiced=%0.2fs seg_mean=%0.3f seg_max=%0.3f topk=%d",
                    name,
                    best_start / self.sample_rate if best_start is not None else 0.0,
                    best_spk if best_spk is not None else -1,
                    best_stats["voiced_sec"],
                    best_stats["seg_mean"],
                    best_stats["seg_max"],
                    min(self._enroll_profile_top_k, len(candidate_embs)),
                )
            log.info(
                "  '%s' enrollment: raw→stored=%.3f, sep-src→stored=%.3f (DF=%s)",
                name, raw_d, sep_d,
                "ON" if self._denoiser_enabled else "OFF",
            )

            reextracted[name] = emb
            log.info("Re-extracted '%s' through %sPixIT-seg→weighted-WeSpeaker (dim=%d)",
                     name, "DF→" if self._denoiser_enabled else "", len(emb))
        return reextracted

    def step(self, waveform: np.ndarray) -> StepResult:
        """Run one pipeline step on a 5 s audio chunk.

        Output audio path: local separated sources are first mapped to global
        speakers for the current chunk, then those chunk-level global sources
        are overlap-added across the recent rolling buffer. The emitted audio is
        the exact current ``step_samples`` slice of that aggregated global
        source, which is more stable than stitching raw tails from one PixIT
        window at a time.

        Parameters
        ----------
        waveform : np.ndarray, shape (chunk_samples,) or (chunk_samples, 1)
            5 s of mono audio at 16 kHz, float32, range [-1, 1].

        Returns
        -------
        StepResult
            Per-speaker audio, activity scores, and timing info.
        """
        t0 = time.perf_counter()
        tc = time.perf_counter
        _si = self._step_idx

        # ── Prepare waveform ──
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform[:, 0]                   # (samples,)
        assert waveform.shape[0] == self.chunk_samples, (
            f"Expected {self.chunk_samples} samples, got {waveform.shape[0]}"
        )

        # ── 1. PixIT: direct GPU call (bypass SpeakerSegmentation formatter) ──
        wav_gpu = torch.from_numpy(waveform).to(
            device=self.device, dtype=torch.float16     # permanent fp16 model
        )
        wav_gpu = wav_gpu.unsqueeze(0).unsqueeze(0)    # (1, 1, 80000)
        t_upload = tc()

        with torch.inference_mode():
            diarization_t = self._pixit_wrapper(wav_gpu)  # (1, 624, 3)  fp16
        if self._step_perf_log and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        t1 = tc()
        seg_np = diarization_t[0].float().cpu().numpy()   # (624, 3) ensure fp32
        sources = self._pixit_wrapper.last_sources.float()  # (1, 80000, 3) GPU fp32

        # One GPU→CPU sync for per-local tail RMS (reconcile dedupe + holdover used to
        # slice each lane and call .cpu() separately → several syncs per step when
        # multiple locals talk → large infer_ms spikes on Jetson).
        with torch.inference_mode():
            step_tail_rms_np = (
                sources[0, -self.step_samples :, :]
                .pow(2)
                .mean(dim=0)
                .sqrt()
                .cpu()
                .numpy()
            )

        t2 = tc()
        # Build SlidingWindowFeature for clustering
        start_time = self._chunk_start_time(self._step_idx)
        n_frames = seg_np.shape[0]
        resolution = self.duration / n_frames
        sw = SlidingWindow(start=start_time, duration=resolution, step=resolution)
        segmentation = SlidingWindowFeature(seg_np, sw)
        embed_waveform, embed_seg = self._embedding_window(waveform, seg_np)

        # ── 2. Embeddings from mixed waveform + segmentation weights ──
        # This is diart's official path: one mixed chunk, one segmentation matrix,
        # one weighted embedding pass for all local speakers.
        active_mask = np.max(embed_seg, axis=0) >= self.cfg["clustering"]["tau_active"]
        n_active_locals = int(np.count_nonzero(active_mask))
        embeddings = torch.full(
            (self.n_local_speakers, self._emb_dim), float("nan"),
            device="cpu",
        )

        active_indices: List[int] = []
        embed_recent_failures = 0
        embed_full_fallback_hits = 0
        source_stats = [
            {
                "peak": 0.0,
                "voiced_sec": 0.0,
                "voiced_ratio": 0.0,
                "seg_max": 0.0,
                "seg_mean": 0.0,
                "tail_rms": 0.0,
                "has_embedding": False,
            }
            for _ in range(self.n_local_speakers)
        ]
        with torch.inference_mode():
            for spk_idx in range(self.n_local_speakers):
                stats = self._source_stats(
                    sources[0, :, spk_idx],
                    seg_np[:, spk_idx],
                    recent_sec=self._embed_recent_sec,
                )
                stats["tail_rms"] = float(step_tail_rms_np[spk_idx])
                source_stats[spk_idx].update(stats)
                if not active_mask[spk_idx]:
                    continue
                if (
                    stats["voiced_sec"] < self._embed_min_voiced_sec
                    and not self.clustering._anchors
                ):
                    embed_recent_failures += 1
                    continue
                active_indices.append(spk_idx)

        t_prep = tc()

        embed_batches = (
            [{"samples": int(embed_waveform.shape[0]), "count": int(len(active_indices))}]
            if active_indices
            else []
        )

        with torch.inference_mode():
            if active_indices:
                embs = self._oa_emb(
                    embed_waveform[np.newaxis, :, np.newaxis],
                    embed_seg[np.newaxis, :, :],
                )
                embs = torch.as_tensor(embs, dtype=torch.float32)
                if embs.ndim == 3 and embs.shape[0] == 1:
                    embs = embs[0]
                if embs.ndim == 1:
                    embs = embs.unsqueeze(0)
                for spk_idx in active_indices:
                    embeddings[spk_idx] = embs[spk_idx].cpu()
                    source_stats[spk_idx]["has_embedding"] = True
        verify_embeddings = self._extract_source_verify_embeddings(sources, active_indices)
        self.clustering.set_verify_embeddings(verify_embeddings.numpy())

        t3 = tc()
        # ── 3. Clustering ──
        permuted_seg = self.clustering(segmentation, embeddings)
        speaker_map = self.clustering.last_speaker_map

        t_cluster = tc()
        # ── 3b. DIART-style delayed aggregation on permuted segmentation ──
        self._pred_seg_buffer.append(permuted_seg)
        agg_permuted = self._pred_aggregation(self._pred_seg_buffer)
        if len(self._pred_seg_buffer) == self._pred_aggregation.num_overlapping_windows:
            self._pred_seg_buffer = self._pred_seg_buffer[1:]

        self._emit_pred_buffer.append(permuted_seg)
        emit_agg_permuted = self._emit_pred_aggregation(self._emit_pred_buffer)
        if len(self._emit_pred_buffer) == self._emit_pred_aggregation.num_overlapping_windows:
            self._emit_pred_buffer = self._emit_pred_buffer[1:]

        emit_step_scores = np.asarray(emit_agg_permuted.data, dtype=np.float32)

        t4 = tc()
        # ── 4. Map sources to global speakers (single window) ──
        # One PixIT forward per step; we only read the current ``sources`` tail.
        # Multi-window overlap-add on *waveforms* was reverted: misaligned
        # windows summed wrong phases → echo/buildup.  DelayedAggregation above
        # only smooths *segmentation* for activity/UI, not separated audio.

        results = []
        pairs: List[tuple[int, int]] = []
        used_local: set[int] = set()
        speaker_map_pairs = 0
        anchor_checks = 0
        remap_count = 0
        dedup_drop_count = 0
        holdover_added = 0
        onset_grace_added = 0
        if speaker_map is not None:
            li, gi = speaker_map.valid_assignments(strict=True)
            speaker_map_pairs = int(len(li))
            for l, g in zip(li, gi):
                l_i, g_i = int(l), int(g)
                if l_i in used_local:
                    continue
                pairs.append((l_i, g_i))
                used_local.add(l_i)

        pairs_before_aux = list(pairs)
        pairs, anchor_checks = self._onset_aux_enrolled_pair(
            pairs,
            active_indices,
            source_stats,
            step_tail_rms_np,
            embeddings,
        )
        remap_count = int(pairs != pairs_before_aux)

        pairs_after_reconcile = int(len(pairs))
        unknown_pairs_dropped = 0

        t5 = tc()
        chunk_global = np.zeros(
            (self.chunk_samples, self.clustering.max_speakers),
            dtype=np.float32,
        )
        emit_pairs: List[tuple[int, int, float, bool]] = []
        local_step_frames = max(
            1, int(round(seg_np.shape[0] * self.step_duration / self.duration))
        )
        if pairs:
            enrolled_pair_globals = [
                int(global_idx)
                for _, global_idx in pairs
                if self.clustering.is_enrolled(int(global_idx))
            ]
            for local_idx, global_idx in pairs:
                activity = float(np.mean(seg_np[-local_step_frames:, local_idx]))
                is_enrolled = self.clustering.is_enrolled(global_idx)
                if (
                    not is_enrolled
                    and enrolled_pair_globals
                    and bool(source_stats[local_idx]["has_embedding"])
                ):
                    emb_row = verify_embeddings[local_idx].numpy().astype(np.float64, copy=False)
                    if np.isnan(emb_row).any():
                        emb_row = embeddings[local_idx].numpy().astype(np.float64, copy=False)
                    if not np.isnan(emb_row).any():
                        nearest_enrolled = min(
                            float(
                                self.clustering.enrolled_distance(
                                    g_idx,
                                    emb_row,
                                    int(local_idx),
                                )
                            )
                            for g_idx in enrolled_pair_globals
                        )
                        if nearest_enrolled <= (self.clustering.delta_enrolled + 0.03):
                            unknown_pairs_dropped += 1
                            continue
                src = sources[0, :, local_idx].detach().cpu().numpy()
                chunk_global[:, global_idx] = src.astype(np.float32, copy=False)
                step_track = seg_np[-local_step_frames:, local_idx]
                emit_pairs.append((local_idx, global_idx, activity, is_enrolled))

        delayed_audio = self._aggregate_audio_chunk(self._step_idx, chunk_global)
        delayed_mix_audio = self._aggregate_mix_chunk(self._step_idx, waveform)
        current_emit_meta: List[Dict[str, Any]] = []
        for local_idx, global_idx, activity, is_enrolled in emit_pairs:
            label = self.clustering.get_label(global_idx)
            id_sim: Optional[float] = None
            if global_idx in self.clustering._anchors:
                _row = verify_embeddings[local_idx].numpy()
                if np.isnan(_row).any():
                    _row = embeddings[local_idx].numpy()
                if not np.isnan(_row).any():
                    _a = self.clustering._anchors[global_idx]
                    id_sim = float(
                        np.clip(np.dot(_row.astype(np.float64), _a), -1.0, 1.0)
                    )
            current_emit_meta.append(
                {
                    "local_idx": int(local_idx),
                    "global_idx": int(global_idx),
                    "label": label,
                    "activity": float(activity),
                    "is_enrolled": bool(is_enrolled),
                    "identity_similarity": id_sim,
                }
            )

        self._emit_meta_buffer.append((self._step_idx, current_emit_meta))
        if len(self._emit_meta_buffer) > self._audio_aggregation.num_overlapping_windows:
            self._emit_meta_buffer = self._emit_meta_buffer[-self._audio_aggregation.num_overlapping_windows :]

        emit_step_idx = self._step_idx - self._output_latency_steps + 1
        delayed_emit_meta: List[Dict[str, Any]] = []
        for hist_step_idx, hist_meta in reversed(self._emit_meta_buffer):
            if hist_step_idx == emit_step_idx:
                delayed_emit_meta = hist_meta
                break

        delayed_meta_by_global: Dict[int, Dict[str, Any]] = {
            int(meta["global_idx"]): meta for meta in delayed_emit_meta
        }
        candidate_globals = set(delayed_meta_by_global.keys())
        if self._enrolled_live_mix and emit_step_scores.ndim == 2:
            continuity_min = max(0.02, self._enrolled_mix_gate_threshold * 0.5)
            for global_idx in self.clustering._anchors:
                if global_idx >= emit_step_scores.shape[1]:
                    continue
                if float(np.max(emit_step_scores[:, global_idx])) >= continuity_min:
                    candidate_globals.add(int(global_idx))

        for global_idx in sorted(candidate_globals):
            meta = delayed_meta_by_global.get(global_idx)
            is_enrolled = bool(meta["is_enrolled"]) if meta is not None else self.clustering.is_enrolled(global_idx)
            if not is_enrolled and meta is None:
                continue
            label = str(meta["label"]) if meta is not None else self.clustering.get_label(global_idx)
            id_sim = meta["identity_similarity"] if meta is not None else None
            if emit_step_scores.ndim == 2 and global_idx < emit_step_scores.shape[1]:
                global_track = emit_step_scores[:, global_idx]
            else:
                global_track = np.zeros(1, dtype=np.float32)
            activity = float(np.mean(global_track)) if global_track.size else 0.0

            audio = delayed_audio[:, global_idx].astype(np.float32, copy=False)
            if is_enrolled and self._enrolled_live_mix:
                mix_mask = self._step_track_mask(
                    global_track,
                    self._enrolled_mix_gate_collar_sec,
                )
                if mix_mask.shape[0] != delayed_mix_audio.shape[0]:
                    mix_mask = np.resize(mix_mask, delayed_mix_audio.shape[0]).astype(np.float32)
                audio = delayed_mix_audio.astype(np.float32, copy=False) * mix_mask
                active_frac = float(np.mean(mix_mask)) if mix_mask.size else 0.0
                if active_frac > 1e-3:
                    audio = audio / max(np.sqrt(active_frac), 0.4)
            if is_enrolled and self._df_output_enrolled:
                audio = self._enhance_enrolled_audio(audio, global_idx)
            audio = np.clip(audio, -1.0, 1.0)
            packet_rms = float(
                np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            ) if audio.size else 0.0
            if packet_rms < 1e-4:
                unknown_pairs_dropped += 1
                continue

            results.append(SpeakerResult(
                global_idx=global_idx,
                label=label,
                audio=audio,
                activity=activity,
                is_enrolled=is_enrolled,
                identity_similarity=id_sim,
            ))

        t6 = tc()
        self._step_idx += 1
        infer_ms = (time.perf_counter() - t0) * 1000
        should_log_perf = (
            self._step_perf_log
            and (
                self._step_perf_every_step
                or _si % 6 == 0
                or infer_ms >= self._step_perf_spike_threshold_ms
            )
        )
        if should_log_perf:
            local_perf = []
            for idx in range(self.n_local_speakers):
                local_perf.append(
                    {
                        "idx": int(idx),
                        "active": bool(active_mask[idx]),
                        "has_embedding": bool(source_stats[idx]["has_embedding"]),
                        "has_verify_embedding": bool(
                            not np.isnan(verify_embeddings[idx].numpy()).any()
                        ),
                        "seg_max": round(float(source_stats[idx]["seg_max"]), 4),
                        "voiced_sec": round(float(source_stats[idx]["voiced_sec"]), 4),
                        "tail_rms": round(float(step_tail_rms_np[idx]), 5),
                    }
                )
            assignment_perf = []
            for local_idx, global_idx in pairs:
                assignment_perf.append(
                    {
                        "local": int(local_idx),
                        "global": int(global_idx),
                        "label": self.clustering.get_label(int(global_idx)),
                    }
                )
            perf_payload = {
                "type": "step_perf",
                "step": int(_si),
                "infer_ms": round(float(infer_ms), 3),
                "timings_ms": {
                    "input_h2d": round((t_upload - t0) * 1000, 3),
                    "pixit": round((t1 - t_upload) * 1000, 3),
                    "seg_tail_sync": round((t2 - t1) * 1000, 3),
                    "embed_prep": round((t_prep - t2) * 1000, 3),
                    "embed_forward": round((t3 - t_prep) * 1000, 3),
                    "cluster": round((t_cluster - t3) * 1000, 3),
                    "aggregate": round((t4 - t_cluster) * 1000, 3),
                    "map": round((t5 - t4) * 1000, 3),
                    "extract": round((t6 - t5) * 1000, 3),
                },
                "counts": {
                    "active_locals": int(n_active_locals),
                    "prepared_sources": int(len(active_indices)),
                    "recent_prepare_failures": int(embed_recent_failures),
                    "full_buffer_fallback_hits": int(embed_full_fallback_hits),
                    "embed_batches": embed_batches,
                    "speaker_map_pairs": int(speaker_map_pairs),
                    "pairs_after_reconcile": int(pairs_after_reconcile),
                    "holdover_added": int(holdover_added),
                    "onset_grace_added": int(onset_grace_added),
                    "unknown_pairs_dropped": int(unknown_pairs_dropped),
                    "final_pairs": int(len(pairs)),
                    "anchor_count": int(len(self.clustering._anchors)),
                    "anchor_checks": int(anchor_checks),
                    "remap_count": int(remap_count),
                    "dedup_drop_count": int(dedup_drop_count),
                },
                "locals": local_perf,
                "assignments": assignment_perf,
                "spike": bool(infer_ms >= self._step_perf_spike_threshold_ms),
            }
            self._perf_log_append(perf_payload)
            log.info(
                "STEP_PERF step=%d infer_ms=%.1fms h2d=%.1f pixit=%.1f seg_tail=%.1f "
                "embed_prep=%.1f embed_fwd=%.1f cluster=%.1f agg=%.1f map=%.1f "
                "extract=%.1f active=%d prep=%d fallback=%d anchors=%d checks=%d "
                "holdover=%d onset=%d pairs=%d",
                _si,
                infer_ms,
                (t_upload - t0) * 1000,
                (t1 - t_upload) * 1000,
                (t2 - t1) * 1000,
                (t_prep - t2) * 1000,
                (t3 - t_prep) * 1000,
                (t_cluster - t3) * 1000,
                (t4 - t_cluster) * 1000,
                (t5 - t4) * 1000,
                (t6 - t5) * 1000,
                n_active_locals,
                len(active_indices),
                embed_full_fallback_hits,
                len(self.clustering._anchors),
                anchor_checks,
                holdover_added,
                onset_grace_added,
                len(pairs),
            )

        return StepResult(
            speakers=results,
            infer_ms=infer_ms,
            step_idx=max(0, emit_step_idx),
        )

    def reset(self) -> None:
        """Reset pipeline state for a new session."""
        self._step_idx = 0
        self._audio_buffer.clear()
        self._mix_buffer.clear()
        self._emit_meta_buffer.clear()
        self._pred_seg_buffer.clear()
        self._emit_pred_buffer.clear()
        self._df_output_states.clear()
        # Re-create clustering with same params, re-inject enrolled centroids
        clust_cfg = self.cfg["clustering"]
        old_anchors = {
            self.clustering._labels[idx]: self.clustering._anchors[idx]
            for idx in self.clustering._anchors
        }
        self.clustering = EnrolledSpeakerClustering(
            tau_active=clust_cfg["tau_active"],
            rho_update=clust_cfg["rho_update"],
            delta_new=clust_cfg["delta_new"],
            metric="cosine",
            max_speakers=clust_cfg["max_speakers"],
            max_drift=clust_cfg["max_drift"],
            delta_enrolled=clust_cfg.get("delta_enrolled", 0.80),
            leakage_delta=clust_cfg.get("leakage_delta"),
        )
        if old_anchors:
            self.clustering.inject_centroids(old_anchors)
            self.clustering.inject_profiles(self._enrollment_profiles_by_name)
