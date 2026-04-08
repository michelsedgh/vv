"""
RealtimePipeline — orchestrates DeepFilterNet → PixIT → WeSpeaker → EnrolledClustering.

Uses PixIT's separated sources for embedding extraction. diart's official
weights-based embedding path applies masks to the original mixed waveform; once
we already have separated sources from PixIT, the correct path is to embed those
fixed-length separated channels directly and batch them efficiently.

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

Playback quality, “choppiness”, and spectrogram gaps (design notes)
──────────────────────────────────────────────────────────────────
Realtime output is one **new 500 ms slice** per global speaker per step, taken
from the **tail** of the current PixIT separated sources (single 5 s window).
That slice is **not** the same physical audio as the previous step’s tail:
PixIT is re-run every step on a sliding buffer, so the separated waveform for
“the same” speaker **changes** at chunk boundaries.  Short crossfades hide
that mismatch; they cannot remove model inconsistency entirely.

**What caused audible choppiness / mid-word spectrogram holes (during tuning):**

1. **Empty speaker map** — Clustering only maps locals whose PixIT frame scores
   pass ``tau_active``.  Scores flicker; valid_assignments() can drop out for a
   step → **no audio packet** for ~500 ms → spectrogram gaps.  **Mitigations:**
   slightly lower ``clustering.tau_active`` so “active” is declared more often;
   **enrolled holdover** in ``step()`` re-attaches enrolled globals to their
   last PixIT channel when seg peak or tail RMS stays above
   ``holdover_seg_max`` / ``holdover_tail_rms`` even if the map would be empty.

2. **Large sample discontinuity at chunk joins** — Notable sample jumps and RMS
   swings at boundaries.  **Mitigations:**
   longer overlap ``audio.output_crossfade_ms`` (default 100 ms vs earlier 15–40
   ms) and a **cosine** crossfade (zero derivative at ends) instead of linear.

3. **DeepFilterNet on separated output** — 500 ms sub-blocks with per-speaker
   streaming state add **gain/level steps** at block edges (pumping).  Default
   ``denoiser.enhance_enrolled_output: false`` keeps playback smooth; turn on
   only if denoised headphones output matters more than continuity.

**What we deliberately do *not* do:** overlap-add of separated waveforms across
multiple PixIT windows was tried and **reverted** — time alignment between
windows is not the same as STFT overlap-add; summed tails produced **echo /
buildup**.  DIART-style aggregation applies to **diarization scores**
(``DelayedAggregation`` on permuted_seg), not to separated waveforms.

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
from scipy.spatial.distance import cdist
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
    activity: float            # mean diarization score over the step window
    is_enrolled: bool
    # WeSpeaker: cosine similarity (dot product) of this step's separated-source
    # embedding vs frozen enrollment anchor — only for enrolled; None otherwise.
    # Orthogonal to ``activity`` (which is PixIT/diarization strength).
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

    - Run PixIT on each 5 s chunk; read ``last_sources`` (separated tracks).
    - Extract WeSpeaker embeddings from separated channels using one fixed-size
      batched forward pass.
    - Feed ``EnrolledSpeakerClustering`` for local→global IDs.
    - Smooth **display / activity** with ``DelayedAggregation`` (Hamming), same
      role as diar's ``pred_aggregation`` — does not fuse waveforms.
    - Build per-step **playback audio**: last ``step_samples + fade`` from each
      mapped source, **cosine crossfade** with previous tail per global speaker.
    - Optional **enrolled holdover** when clustering drops the map but energy
      on the persisted channel suggests speech still present.

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

        # Identity uses diart's official path: one mixed waveform chunk plus
        # per-speaker segmentation weights. PixIT-separated channels are used
        # only for final audio extraction, not for verification.
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
        # Keep continuity simple: trust clustering first, then allow a very short
        # enrolled-only continuity rescue on the persisted local channel.
        self._holdover_max_steps = int(clust_cfg.get("holdover_max_steps", 3))
        self._reentry_max_steps = int(clust_cfg.get("reentry_max_steps", 8))
        self._reentry_min_voiced_sec = float(
            clust_cfg.get("reentry_min_voiced_sec", 0.25)
        )
        self._reentry_tail_rms = float(clust_cfg.get("reentry_tail_rms", 0.02))
        self._reentry_dominance_ratio = float(
            clust_cfg.get("reentry_dominance_ratio", 3.0)
        )
        self._unknown_output_min_activity = float(
            clust_cfg.get("unknown_output_min_activity", 0.05)
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
            enrolled_grace_margin=clust_cfg.get("enrolled_grace_margin", 0.05),
            leakage_delta=clust_cfg.get("leakage_delta"),
            new_center_grace_margin=clust_cfg.get("new_center_grace_margin", 0.05),
            enrolled_preference_margin=clust_cfg.get("enrolled_preference_margin", 0.03),
            unknown_reuse_delta=clust_cfg.get("unknown_reuse_delta"),
            unknown_min_voiced_sec=clust_cfg.get("unknown_min_voiced_sec", 0.18),
            unknown_min_tail_rms=clust_cfg.get("unknown_min_tail_rms", 0.0),
        )
        if enrolled_embeddings:
            # Re-extract through EXACT same path as live: [DF→]PixIT→WeSpeaker
            reextracted = self._reextract_enrollments(enrolled_embeddings, config)
            self.clustering.inject_centroids(reextracted)
            log.info(
                "Injected %d enrolled speaker(s): %s",
                len(reextracted),
                ", ".join(reextracted.keys()),
            )

        # ── State ──
        self._step_idx = 0

        # Per-global-speaker crossfade (see module docstring: “Playback quality…”).
        # Each step we emit step_samples of audio; the first fade_samples overlap
        # with the previous step’s tail so chunk boundaries are not hard cuts.
        # Config: audio.output_crossfade_ms (default 100). Capped so we still
        # advance mostly “new” audio each step; floored ~10 ms for stability.
        acfg = config.get("audio", {})
        _fade_ms = float(acfg.get("output_crossfade_ms", 100.0))
        if self._df_output_enrolled:
            _fade_ms = max(_fade_ms, 50.0)  # DF on output adds edge dynamics; keep min overlap
        self._fade_samples = min(
            int((_fade_ms / 1000.0) * self.sample_rate),
            self.step_samples - 400,  # keep a floor of usable new audio per step
        )
        self._fade_samples = max(self._fade_samples, 160)  # at least ~10 ms
        self._prev_tails: Dict[int, np.ndarray] = {}  # global_idx → last fade_samples after processing
        self._prev_tail_step: Dict[int, int] = {}  # global_idx → step_idx when tail last written
        # After this many steps without emitting a global, drop stored tail (avoid blending
        # new speech with a stale or near-silent tail → quiet attack on first word).
        self._crossfade_stale_steps = int(
            acfg.get("crossfade_stale_silent_steps", 6)
        )
        self._xfade_onset_prev_rms = float(
            acfg.get("crossfade_onset_skip_if_prev_below", 0.02)
        )
        self._xfade_onset_new_rms = float(
            acfg.get("crossfade_onset_skip_if_new_above", 0.025)
        )
        log.info(
            "Separated-audio crossfade: %d samples (%.1f ms), cosine ramp",
            self._fade_samples,
            1000.0 * self._fade_samples / self.sample_rate,
        )
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
        for name in enrolled_embeddings:
            wav_path = enroll_dir / name / "reference.wav"
            if not wav_path.exists():
                log.warning("No reference.wav for '%s', using stored embedding", name)
                reextracted[name] = enrolled_embeddings[name]
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

                    embed_chunk, embed_seg = self._embedding_window(chunk, seg_np)
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

                    emb_t = self._oa_emb(
                        embed_chunk[np.newaxis, :, np.newaxis],
                        embed_seg[np.newaxis, :, :],
                    )
                    emb_np = np.asarray(emb_t, dtype=np.float64)
                    if emb_np.ndim == 3 and emb_np.shape[0] == 1:
                        emb_np = emb_np[0]
                    if emb_np.ndim == 1:
                        emb_np = emb_np.reshape(1, -1)
                    for spk_idx, stats in batch_meta:
                        emb = emb_np[spk_idx]
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
                prof = np.stack([candidate_embs[i] for i in keep], axis=0).mean(axis=0)
                prof_norm = np.linalg.norm(prof)
                if prof_norm > 0:
                    prof = prof / prof_norm
                emb = prof.astype(np.float64)
            else:
                emb = raw_emb

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

        Output audio path (continuity): ``valid_assignments()`` gives (local,
        global) pairs.  Enrolled **holdover** may append pairs if an enrolled
        speaker disappeared from the map but PixIT still shows energy on the
        last channel we used for them — avoids 500 ms silence holes.

        For each pair we take the last ``step_samples + fade`` samples from
        ``sources`` for that local index, crossfade the first ``fade`` with
        ``_prev_tails[global]`` (cosine ramp), optionally run output DF, store
        new tail, then clip to ``step_samples`` for the client.

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
        start_time = self._step_idx * self.step_duration
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

        t3 = tc()
        # ── 3. Clustering ──
        self.clustering._current_source_stats = source_stats
        permuted_seg = self.clustering(segmentation, embeddings)
        self.clustering._current_source_stats = None
        speaker_map = self.clustering.last_speaker_map

        t_cluster = tc()
        # ── 3b. DIART-style delayed aggregation on permuted segmentation ──
        self._pred_seg_buffer.append(permuted_seg)
        agg_permuted = self._pred_aggregation(self._pred_seg_buffer)
        if len(self._pred_seg_buffer) == self._pred_aggregation.num_overlapping_windows:
            self._pred_seg_buffer = self._pred_seg_buffer[1:]

        n_seg_frames = agg_permuted.data.shape[0]
        step_frames = max(1, int(n_seg_frames * self.step_duration / self.duration))
        step_seg = agg_permuted.data[-step_frames:]

        t4 = tc()
        # ── 4. Map sources to global speakers (single window) ──
        # One PixIT forward per step; we only read the current ``sources`` tail.
        # Multi-window overlap-add on *waveforms* was reverted: misaligned
        # windows summed wrong phases → echo/buildup.  DelayedAggregation above
        # only smooths *segmentation* for activity/UI, not separated audio.

        results = []
        clust_cfg = self.cfg["clustering"]
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

        ho_max = float(clust_cfg.get("holdover_seg_max", 0.12))
        ho_rms = float(clust_cfg.get("holdover_tail_rms", 0.002))

        # Keep only one conservative enrolled continuity rule:
        # if clustering misses the enrolled speaker on the previously used local
        # channel, recover it only when the current embedding is still near the
        # enrolled anchor, or for one immediate step when segmentation and tail
        # energy both still show speech. This avoids silence leakage while still
        # covering weak sentence onsets and brief assignment misses.
        if self.clustering._anchors:
            emb_np = embeddings.detach().cpu().numpy()
            for g_idx in sorted(self.clustering._anchors.keys()):
                if any(g == g_idx for _, g in pairs):
                    continue
                last_seen = self.clustering._last_enrolled_step.get(g_idx, -999)
                gap = int(self.clustering._call_count - last_seen)
                max_continuity_gap = max(
                    int(self._holdover_max_steps),
                    int(self._reentry_max_steps),
                )
                if max_continuity_gap > 0 and gap > max_continuity_gap:
                    continue
                loc = self.clustering._last_local_for_enrolled.get(g_idx)
                if loc is None or loc in used_local:
                    continue
                smax = float(np.max(seg_np[:, loc]))
                t_rms = float(step_tail_rms_np[int(loc)])
                has_embedding = bool(source_stats[loc]["has_embedding"])
                rescue_with_embedding = False
                if has_embedding and not np.isnan(emb_np[loc]).any():
                    anchor_checks += 1
                    anchor = np.asarray(
                        self.clustering._anchors[g_idx], dtype=np.float64
                    ).reshape(1, -1)
                    dist = float(
                        cdist(
                            emb_np[loc : loc + 1].astype(np.float64),
                            anchor,
                            metric="cosine",
                        )[0, 0]
                    )
                    eff = self.clustering._effective_enrolled_delta(g_idx, loc)
                    rescue_with_embedding = (
                        dist < (eff + self.clustering.new_center_grace_margin)
                    )

                cur_pair_idx = next(
                    (i for i, (l_i, _) in enumerate(pairs) if l_i == loc),
                    None,
                )
                if (
                    cur_pair_idx is not None
                    and not self.clustering.is_enrolled(int(pairs[cur_pair_idx][1]))
                    and rescue_with_embedding
                ):
                    pairs[cur_pair_idx] = (loc, g_idx)
                    remap_count += 1
                    onset_grace_added += 1
                    continue

                if cur_pair_idx is not None:
                    continue

                if rescue_with_embedding:
                    pairs.append((loc, g_idx))
                    used_local.add(loc)
                    onset_grace_added += 1
                    continue

                second_tail = 0.0
                if active_indices:
                    active_tail = sorted(
                        (float(step_tail_rms_np[idx]) for idx in active_indices),
                        reverse=True,
                    )
                    if len(active_tail) > 1:
                        second_tail = active_tail[1]
                dominant_reentry = (
                    len(active_indices) == 1
                    or t_rms >= max(
                        self._reentry_tail_rms,
                        second_tail * self._reentry_dominance_ratio,
                    )
                )
                if (
                    gap <= self._reentry_max_steps
                    and active_mask[loc]
                    and source_stats[loc]["voiced_sec"] >= self._reentry_min_voiced_sec
                    and t_rms >= self._reentry_tail_rms
                    and dominant_reentry
                ):
                    if cur_pair_idx is not None:
                        pairs[cur_pair_idx] = (loc, g_idx)
                    else:
                        pairs.append((loc, g_idx))
                        used_local.add(loc)
                    onset_grace_added += 1
                    continue

                if gap <= 1 and smax >= ho_max and t_rms >= ho_rms:
                    pairs.append((loc, g_idx))
                    used_local.add(loc)
                    holdover_added += 1

        pairs_after_reconcile = int(len(pairs))
        unknown_pairs_dropped = 0

        t5 = tc()
        if pairs:
            fade = self._fade_samples
            extract_len = self.step_samples + fade
            local_ids = [p[0] for p in pairs]
            global_ids = [p[1] for p in pairs]
            tail_sources = (
                sources[0, -extract_len:, local_ids]
                .detach()
                .cpu()
                .numpy()
            )

            for out_idx, (local_idx, global_idx) in enumerate(zip(local_ids, global_ids)):
                label = self.clustering.get_label(global_idx)
                activity = float(np.mean(step_seg[:, global_idx]))
                if (
                    not self.clustering.is_enrolled(global_idx)
                    and activity < self._unknown_output_min_activity
                ):
                    unknown_pairs_dropped += 1
                    continue

                id_sim: Optional[float] = None
                if global_idx in self.clustering._anchors:
                    _row = embeddings[local_idx].numpy()
                    if not np.isnan(_row).any():
                        _a = self.clustering._anchors[global_idx]
                        id_sim = float(
                            np.clip(np.dot(_row.astype(np.float64), _a), -1.0, 1.0)
                        )

                raw = tail_sources[:, out_idx].copy()

                prev_tail = self._prev_tails.get(global_idx)
                gap_steps = self._step_idx - self._prev_tail_step.get(
                    global_idx, -(10**9)
                )
                if gap_steps > self._crossfade_stale_steps:
                    prev_tail = None
                do_xfade = False
                if prev_tail is not None and len(prev_tail) == fade:
                    prev_e = float(
                        np.sqrt(np.mean(prev_tail.astype(np.float64) ** 2))
                    )
                    new_e = float(
                        np.sqrt(np.mean(raw[:fade].astype(np.float64) ** 2))
                    )
                    # Cosine crossfade starts at 100% prev_tail → if prev is silence after
                    # an idle gap, the first ~fade ms of new speech is ducked (quiet attack).
                    if (
                        prev_e < self._xfade_onset_prev_rms
                        and new_e > self._xfade_onset_new_rms
                    ):
                        prev_tail = None
                    else:
                        do_xfade = True
                if do_xfade and prev_tail is not None:
                    # Cosine crossfade: ramp = 0.5 - 0.5*cos(0..pi).  Ends have
                    # zero slope vs linear ramp → less zipper / click at joins
                    # when PixIT’s separated tail jumps between steps.
                    t = np.linspace(0.0, np.pi, fade, dtype=np.float32)
                    ramp = 0.5 - 0.5 * np.cos(t)
                    raw[:fade] = prev_tail * (1.0 - ramp) + raw[:fade] * ramp

                # DF after crossfade so block boundaries are not misaligned with fade region
                if self.clustering.is_enrolled(global_idx) and self._df_output_enrolled:
                    raw = self._enhance_enrolled_audio(raw, global_idx)

                self._prev_tails[global_idx] = raw[-fade:].copy()
                self._prev_tail_step[global_idx] = self._step_idx
                audio = np.clip(raw[:self.step_samples], -1.0, 1.0)

                results.append(SpeakerResult(
                    global_idx=global_idx,
                    label=label,
                    audio=audio,
                    activity=activity,
                    is_enrolled=self.clustering.is_enrolled(global_idx),
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
            step_idx=self._step_idx,
        )

    def reset(self) -> None:
        """Reset pipeline state for a new session."""
        self._step_idx = 0
        self._prev_tails.clear()
        self._prev_tail_step.clear()
        self._pred_seg_buffer.clear()
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
            enrolled_grace_margin=clust_cfg.get("enrolled_grace_margin", 0.05),
            leakage_delta=clust_cfg.get("leakage_delta"),
            new_center_grace_margin=clust_cfg.get("new_center_grace_margin", 0.05),
            enrolled_preference_margin=clust_cfg.get("enrolled_preference_margin", 0.03),
            unknown_reuse_delta=clust_cfg.get("unknown_reuse_delta"),
            unknown_min_voiced_sec=clust_cfg.get("unknown_min_voiced_sec", 0.18),
            unknown_min_tail_rms=clust_cfg.get("unknown_min_tail_rms", 0.0),
        )
        if old_anchors:
            self.clustering.inject_centroids(old_anchors)
