"""
RealtimePipeline — orchestrates DeepFilterNet → PixIT → WeSpeaker → EnrolledClustering.

Uses PixIT's separated sources for embedding extraction, but only from the
speech-active portions of each source track. Feeding the full 5 s separated
lane into WeSpeaker let long silence and residual noise dominate identity, and
the enrolled-priority clustering then latched onto the wrong speaker label.

Data flow per step
──────────────────
  waveform np.ndarray (80000,) — 5 s at 16 kHz
    → [optional] DeepFilterNet denoise (16k→48k→denoise→16k)
    → PixIT                → diarization SlidingWindowFeature (624, 3)
                           → sources tensor (1, 80000, 3)
    → WeSpeaker on speech-active separated sources → (3, emb_dim) embeddings
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

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
import yaml
from pyannote.audio import Model as PyanModel
from pyannote.core import SlidingWindow, SlidingWindowFeature

from diart.blocks.aggregation import DelayedAggregation

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
    - Extract WeSpeaker embeddings from **speech-active regions** of separated
      channels (not the raw mix, and not the full noisy 5 s lane).
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

        # Speaker identity should come from speech-bearing samples only.
        # Embedding the entire 5 s separated lane made long silence and room
        # noise dominate the representation.
        self._embed_seg_threshold = float(
            emb_cfg.get("source_seg_threshold", max(0.12, clust_cfg["tau_active"] * 0.5))
        )
        self._embed_min_voiced_sec = float(emb_cfg.get("source_min_voiced_sec", 0.04))
        self._embed_pad_min_sec = float(emb_cfg.get("source_pad_min_sec", 0.50))
        self._embed_context_sec = float(emb_cfg.get("source_context_sec", 0.12))
        self._embed_live_recent_sec = float(emb_cfg.get("source_live_recent_sec", 1.25))
        self._embed_min_voiced_samples = max(
            1, int(self._embed_min_voiced_sec * self.sample_rate)
        )
        self._embed_pad_min_samples = max(
            self._embed_min_voiced_samples,
            int(self._embed_pad_min_sec * self.sample_rate),
        )
        self._embed_context_samples = max(
            0, int(self._embed_context_sec * self.sample_rate)
        )
        self._embed_live_recent_samples = max(
            self._embed_pad_min_samples,
            int(self._embed_live_recent_sec * self.sample_rate),
        )
        # Holdover: after enrolled match, bridge gaps when clustering drops the map.
        # Logs showed ``holdover_max_steps: 3`` blocked *all* no-embedding holdover
        # after ~1.5 s idle (HOLDOVER_BLOCKED_IDLE) while seg still showed activity
        # but recent-window prep returned no embedding (STEP_PAIRS: act true, emb false).
        self._holdover_max_steps = int(clust_cfg.get("holdover_max_steps", 3))
        self._holdover_no_emb_max_steps = int(
            clust_cfg.get("holdover_no_embedding_max_steps", 0)
        )  # 0 = no idle cap when prev channel has no embedding (see holdover loop)

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
            unknown_reuse_delta=clust_cfg.get("unknown_reuse_delta"),
            unknown_min_voiced_sec=clust_cfg.get("unknown_min_voiced_sec", 0.18),
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
        self._unknown_emit_min_seg_max = float(
            clust_cfg.get("unknown_emit_min_seg_max", 0.36)
        )
        log.info(
            "Separated-audio crossfade: %d samples (%.1f ms), cosine ramp",
            self._fade_samples,
            1000.0 * self._fade_samples / self.sample_rate,
        )

        # Warmup both models + detect embedding dim
        log.info("Warming up PixIT + WeSpeaker...")
        dummy_fp16 = torch.randn(1, 1, self.chunk_samples, device=self.device, dtype=torch.float16)
        dummy_fp32 = dummy_fp16.float()
        with torch.inference_mode():
            self._pixit_wrapper(dummy_fp16)       # PixIT is fp16
            d = self._emb_model(dummy_fp32)       # WeSpeaker stays fp32
            self._emb_dim = d.shape[-1]
            # Batch-warmup (3 sources at once)
            self._emb_model(dummy_fp32.expand(self.n_local_speakers, -1, -1))
        torch.cuda.synchronize(self.device)
        log.info("Pipeline ready (emb_dim=%d).", self._emb_dim)

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

        Used for enrollment and for per-source live extraction — guarantees
        identical model + preprocessing for both paths.

        Parameters
        ----------
        audio : np.ndarray, shape (samples,), float32, 16 kHz

        Returns
        -------
        np.ndarray, shape (emb_dim,), float64, L2-normalized
        """
        audio = np.asarray(audio, dtype=np.float32).flatten()
        # Peak-normalize so WeSpeaker sees consistent amplitude
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak
        wav = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._emb_model(wav)   # (1, emb_dim)
        emb = emb.squeeze(0).cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb

    def _seg_to_sample_weights(
        self,
        seg_track: np.ndarray,
        num_samples: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Upsample one diarization track from frames to samples."""
        weights = np.asarray(seg_track, dtype=np.float32).reshape(-1)
        if weights.size == 0:
            return torch.zeros(num_samples, device=device, dtype=torch.float32)
        if weights.size == 1:
            return torch.full(
                (num_samples,),
                float(weights[0]),
                device=device,
                dtype=torch.float32,
            )
        xp = np.linspace(0, weights.size - 1, num=num_samples, dtype=np.float32)
        sample_weights = np.interp(
            xp,
            np.arange(weights.size, dtype=np.float32),
            weights,
        ).astype(np.float32)
        return torch.from_numpy(sample_weights).to(device=device, dtype=torch.float32)

    def _prepare_source_for_embedding(
        self,
        source_audio: torch.Tensor,
        seg_track: np.ndarray,
        focus_recent_samples: Optional[int] = None,
    ) -> tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Keep only speech-active samples from a separated source before embedding."""
        src = source_audio.float().flatten()
        if focus_recent_samples is not None and focus_recent_samples > 0 and src.numel() > focus_recent_samples:
            src = src[-focus_recent_samples:]
        peak = float(src.abs().max())
        seg_arr = np.asarray(seg_track, dtype=np.float32).reshape(-1)
        if focus_recent_samples is not None and focus_recent_samples > 0 and len(seg_arr) > 0:
            keep_ratio = min(1.0, focus_recent_samples / float(max(1, source_audio.numel())))
            keep_frames = max(1, int(np.ceil(len(seg_arr) * keep_ratio)))
            seg_arr = seg_arr[-keep_frames:]
        stats = {
            "peak": peak,
            "voiced_sec": 0.0,
            "voiced_ratio": 0.0,
            "seg_max": float(np.max(seg_arr)) if len(seg_arr) else 0.0,
            "seg_mean": float(np.mean(seg_arr)) if len(seg_arr) else 0.0,
        }
        if peak < 1e-4:
            return None, stats

        sample_weights = self._seg_to_sample_weights(seg_arr, src.numel(), src.device)
        speech_mask = sample_weights >= self._embed_seg_threshold
        if self._embed_context_samples > 0 and bool(speech_mask.any()):
            kernel = 2 * self._embed_context_samples + 1
            expanded = F.max_pool1d(
                speech_mask.float().view(1, 1, -1),
                kernel_size=kernel,
                stride=1,
                padding=self._embed_context_samples,
            )
            speech_mask = expanded.view(-1) > 0
        voiced_count = int(speech_mask.sum().item())
        stats["voiced_sec"] = voiced_count / float(self.sample_rate)
        stats["voiced_ratio"] = voiced_count / float(max(1, src.numel()))
        if voiced_count < self._embed_min_voiced_samples:
            return None, stats

        voiced = src[speech_mask] * sample_weights[speech_mask]
        voiced_peak = float(voiced.abs().max())
        if voiced_peak < 1e-4:
            return None, stats
        voiced = voiced / voiced_peak
        if voiced.numel() < self._embed_pad_min_samples:
            voiced = F.pad(voiced, (0, self._embed_pad_min_samples - voiced.numel()))
        return voiced.unsqueeze(0).unsqueeze(0), stats

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

            best_emb: Optional[np.ndarray] = None
            best_stats: Optional[Dict[str, float]] = None
            best_spk: Optional[int] = None
            best_start: Optional[int] = None

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

                    for spk_idx in range(seg_np.shape[1]):
                        prepared, stats = self._prepare_source_for_embedding(
                            sources[0, :, spk_idx],
                            seg_np[:, spk_idx],
                        )
                        if prepared is None:
                            continue
                        emb_t = self._emb_model(prepared)
                        emb_t = emb_t / emb_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        emb = emb_t.squeeze(0).cpu().numpy().astype(np.float64)
                        score = (
                            stats["voiced_sec"],
                            stats["seg_mean"],
                            stats["seg_max"],
                            stats["peak"],
                        )
                        best_score = (
                            best_stats["voiced_sec"],
                            best_stats["seg_mean"],
                            best_stats["seg_max"],
                            best_stats["peak"],
                        ) if best_stats is not None else None
                        if best_score is None or score > best_score:
                            best_emb = emb
                            best_stats = stats
                            best_spk = spk_idx
                            best_start = start

            raw_emb = self.extract_embedding(audio)
            emb = best_emb if best_emb is not None else raw_emb

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
                    "  '%s' PixIT chunk=%0.2fs local=%d voiced=%0.2fs seg_mean=%0.3f seg_max=%0.3f",
                    name,
                    best_start / self.sample_rate if best_start is not None else 0.0,
                    best_spk if best_spk is not None else -1,
                    best_stats["voiced_sec"],
                    best_stats["seg_mean"],
                    best_stats["seg_max"],
                )
            log.info(
                "  '%s' enrollment: raw→stored=%.3f, sep-src→stored=%.3f (DF=%s)",
                name, raw_d, sep_d,
                "ON" if self._denoiser_enabled else "OFF",
            )

            reextracted[name] = emb
            log.info("Re-extracted '%s' through %sPixIT-sep→WeSpeaker (dim=%d)",
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

        with torch.inference_mode():
            diarization_t = self._pixit_wrapper(wav_gpu)  # (1, 624, 3)  fp16

        seg_np = diarization_t[0].float().cpu().numpy()   # (624, 3) ensure fp32
        sources = self._pixit_wrapper.last_sources.float()  # (1, 80000, 3) GPU fp32

        # Build SlidingWindowFeature for clustering
        start_time = self._step_idx * self.step_duration
        n_frames = seg_np.shape[0]
        resolution = self.duration / n_frames
        sw = SlidingWindow(start=start_time, duration=resolution, step=resolution)
        segmentation = SlidingWindowFeature(seg_np, sw)

        # ── 2. Embeddings from speech-active separated sources ──
        # PixIT separation helps, but identity still degrades if we embed the
        # whole 5 s lane because most of that lane can be silence or residual
        # room noise. Restrict embeddings to speech-active samples so identity
        # follows voice rather than background.
        active_mask = np.max(seg_np, axis=0) >= self.cfg["clustering"]["tau_active"]
        embeddings = torch.full(
            (self.n_local_speakers, self._emb_dim), float("nan"),
            device="cpu",
        )

        prepared_sources = []
        source_stats = [
            {
                "peak": 0.0,
                "voiced_sec": 0.0,
                "voiced_ratio": 0.0,
                "seg_max": 0.0,
                "seg_mean": 0.0,
                "has_embedding": False,
            }
            for _ in range(self.n_local_speakers)
        ]
        with torch.inference_mode():
            for spk_idx in range(self.n_local_speakers):
                if not active_mask[spk_idx]:
                    continue
                prepared, stats = self._prepare_source_for_embedding(
                    sources[0, :, spk_idx],
                    seg_np[:, spk_idx],
                    focus_recent_samples=self._embed_live_recent_samples,
                )
                source_stats[spk_idx].update(stats)
                # Recent window (~1.25 s) can be silent while full 5 s chunk still has
                # speech (logs: act true, smx high, emb false). Retry full buffer once.
                if prepared is None:
                    prepared, stats_fb = self._prepare_source_for_embedding(
                        sources[0, :, spk_idx],
                        seg_np[:, spk_idx],
                        focus_recent_samples=None,
                    )
                    if prepared is not None:
                        source_stats[spk_idx].update(stats_fb)
                if prepared is None:
                    continue
                source_stats[spk_idx]["has_embedding"] = True
                prepared_sources.append((spk_idx, prepared))

            for spk_idx, prepared in prepared_sources:
                emb = self._emb_model(prepared)  # (1, emb_dim)
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                embeddings[spk_idx] = emb.squeeze(0).cpu()

        # ── 3. Clustering ──
        self.clustering._current_source_stats = source_stats
        permuted_seg = self.clustering(segmentation, embeddings)
        self.clustering._current_source_stats = None
        speaker_map = self.clustering.last_speaker_map

        # ── 3b. DIART-style delayed aggregation on permuted segmentation ──
        self._pred_seg_buffer.append(permuted_seg)
        agg_permuted = self._pred_aggregation(self._pred_seg_buffer)
        if len(self._pred_seg_buffer) == self._pred_aggregation.num_overlapping_windows:
            self._pred_seg_buffer = self._pred_seg_buffer[1:]

        n_seg_frames = agg_permuted.data.shape[0]
        step_frames = max(1, int(n_seg_frames * self.step_duration / self.duration))
        step_seg = agg_permuted.data[-step_frames:]

        # ── 4. Map sources to global speakers (single window) ──
        # One PixIT forward per step; we only read the current ``sources`` tail.
        # Multi-window overlap-add on *waveforms* was reverted: misaligned
        # windows summed wrong phases → echo/buildup.  DelayedAggregation above
        # only smooths *segmentation* for activity/UI, not separated audio.

        results = []
        clust_cfg = self.cfg["clustering"]
        pairs: List[tuple[int, int]] = []
        used_local: set[int] = set()
        if speaker_map is not None:
            li, gi = speaker_map.valid_assignments()
            for l, g in zip(li, gi):
                l_i, g_i = int(l), int(g)
                if l_i in used_local:
                    continue
                pairs.append((l_i, g_i))
                used_local.add(l_i)

        # SpeakerMap uses Hungarian assignment → at most one local per global column.
        # PixIT often activates 2–3 locals for one voice; the extra lanes then map to
        # Unknown-* even when their embeddings are within delta_enrolled (see logs:
        # STEP_PAIRS with [[0,2],[2,1]] while dist_min to anchor is just above threshold
        # for the “winner” only).  Re-attach any embedded local within the enrolled
        # radius to the anchor, then keep one stream per enrolled global (loudest tail).
        if self.clustering._anchors:
            from scipy.spatial.distance import cdist

            emb_np = embeddings.detach().cpu().numpy()
            _remap_meta: List[dict] = []

            def _tail_rms(lc: int) -> float:
                return float(
                    sources[0, -self.step_samples :, lc]
                    .float()
                    .pow(2)
                    .mean()
                    .sqrt()
                    .cpu()
                )

            pair_list = list(pairs)
            loc_to_g = {l: g for l, g in pair_list}

            for g_anchor in sorted(self.clustering._anchors.keys()):
                a = np.asarray(
                    self.clustering._anchors[g_anchor], dtype=np.float64
                ).reshape(1, -1)
                for loc in range(self.n_local_speakers):
                    if not source_stats[loc]["has_embedding"]:
                        continue
                    if np.isnan(emb_np[loc]).any():
                        continue
                    d = float(
                        cdist(
                            emb_np[loc : loc + 1].astype(np.float64),
                            a,
                            metric="cosine",
                        )[0, 0]
                    )
                    eff = self.clustering._effective_enrolled_delta(g_anchor, loc)
                    if d >= eff:
                        continue
                    g_cur = loc_to_g.get(loc)
                    if g_cur == g_anchor:
                        continue
                    if g_cur is not None and self.clustering.is_enrolled(int(g_cur)):
                        continue
                    if g_cur is not None:
                        pair_list = [(l, g) for l, g in pair_list if l != loc]
                        _remap_meta.append(
                            {
                                "loc": int(loc),
                                "from_g": int(g_cur),
                                "to_g": int(g_anchor),
                                "d": round(d, 4),
                            }
                        )
                    pair_list.append((loc, g_anchor))
                    loc_to_g[loc] = g_anchor

            by_g: Dict[int, List[int]] = {}
            for l, g in pair_list:
                by_g.setdefault(int(g), []).append(int(l))
            deduped: List[tuple[int, int]] = []
            _drop_meta: List[dict] = []
            for g, locs in by_g.items():
                if len(locs) == 1:
                    deduped.append((locs[0], g))
                    continue
                if self.clustering.is_enrolled(g):
                    best = max(locs, key=_tail_rms)
                    for lc in locs:
                        if lc != best:
                            _drop_meta.append(
                                {
                                    "loc": int(lc),
                                    "g": int(g),
                                    "kept": int(best),
                                }
                            )
                    deduped.append((best, g))
                else:
                    for lc in locs:
                        deduped.append((lc, g))
            pairs = deduped
            used_local = {l for l, _ in pairs}

            # #region agent log
            if _remap_meta or _drop_meta:
                import json as _j
                import time as _t

                open(
                    "/home/michel/Documents/Voice/.cursor/debug-21cffc.log", "a"
                ).write(
                    _j.dumps(
                        {
                            "sessionId": "21cffc",
                            "hypothesisId": "H6",
                            "location": "pipeline.py:enroll_reconcile",
                            "message": "ENROLL_REMAP",
                            "data": {
                                "step": self._step_idx,
                                "remaps": _remap_meta,
                                "enrolled_dedup_drops": _drop_meta,
                            },
                            "timestamp": int(_t.time() * 1000),
                        }
                    )
                    + "\n"
                )
            # #endregion

        # Enrolled holdover: clustering uses max(seg)>=tau_active; PixIT scores flicker
        # frame-to-frame so the map can be empty while the separated channel still has
        # speech → 500 ms audio gaps.  Re-attach enrolled globals to their persisted
        # local channel when seg peak or tail RMS says energy is still there.
        ho_max = float(clust_cfg.get("holdover_seg_max", 0.12))
        ho_rms = float(clust_cfg.get("holdover_tail_rms", 0.002))
        if self.clustering._anchors:
            for g_idx in sorted(self.clustering._anchors.keys()):
                if any(g == g_idx for _, g in pairs):
                    continue
                last_seen = self.clustering._last_enrolled_step.get(g_idx, -999)
                gap = int(self.clustering._call_count - last_seen)
                loc = self.clustering._last_local_for_enrolled.get(g_idx)
                if loc is None or loc in used_local:
                    continue
                if source_stats[loc]["has_embedding"]:
                    # #region agent log
                    if self._step_idx % 4 == 0:
                        import json as _j
                        import time as _t

                        open(
                            "/home/michel/Documents/Voice/.cursor/debug-21cffc.log",
                            "a",
                        ).write(
                            _j.dumps(
                                {
                                    "sessionId": "21cffc",
                                    "hypothesisId": "H5",
                                    "location": "pipeline.py:holdover",
                                    "message": "HOLDOVER_SKIP_HAS_EMB",
                                    "data": {
                                        "step": self._step_idx,
                                        "global": int(g_idx),
                                        "loc": int(loc),
                                        "in_pairs_already": any(
                                            g == g_idx for _, g in pairs
                                        ),
                                    },
                                    "timestamp": int(_t.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                    # #endregion
                    continue
                if (
                    self._holdover_no_emb_max_steps > 0
                    and gap > self._holdover_no_emb_max_steps
                ):
                    # #region agent log
                    if self._step_idx % 4 == 0:
                        import json as _j
                        import time as _t

                        open(
                            "/home/michel/Documents/Voice/.cursor/debug-21cffc.log",
                            "a",
                        ).write(
                            _j.dumps(
                                {
                                    "sessionId": "21cffc",
                                    "hypothesisId": "H4",
                                    "location": "pipeline.py:holdover",
                                    "message": "HOLDOVER_BLOCKED_IDLE",
                                    "data": {
                                        "step": self._step_idx,
                                        "global": int(g_idx),
                                        "steps_since_match": gap,
                                        "hmax": int(self._holdover_no_emb_max_steps),
                                        "path": "no_embedding_cap",
                                    },
                                    "timestamp": int(_t.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                    # #endregion
                    continue
                smax = float(np.max(seg_np[:, loc]))
                t_rms = float(
                    sources[0, -self.step_samples :, loc].float().pow(2).mean().sqrt().cpu()
                )
                if smax < ho_max and t_rms < ho_rms:
                    continue
                pairs.append((loc, g_idx))
                used_local.add(loc)
                # #region agent log
                if self._step_idx % 2 == 0:
                    import json as _j
                    import time as _t

                    open(
                        "/home/michel/Documents/Voice/.cursor/debug-21cffc.log", "a"
                    ).write(
                        _j.dumps(
                            {
                                "sessionId": "21cffc",
                                "hypothesisId": "H2",
                                "location": "pipeline.py:holdover",
                                "message": "HOLDOVER_ADD",
                                "data": {
                                    "step": self._step_idx,
                                    "global": int(g_idx),
                                    "loc": int(loc),
                                    "smax": round(smax, 4),
                                    "tail_rms": round(t_rms, 5),
                                    "steps_since_enrolled_match": int(
                                        self.clustering._call_count
                                        - self.clustering._last_enrolled_step.get(
                                            g_idx, -999
                                        )
                                    ),
                                },
                                "timestamp": int(_t.time() * 1000),
                            }
                        )
                        + "\n"
                    )
                # #endregion

        # Drop weak Unknown-* streams: logs showed [[2,1]] with smx≈0.29 right at
        # tau_active (0.28) — phantom lanes before real speech. Enrolled unaffected.
        if self._unknown_emit_min_seg_max > 0:
            _pairs_before = len(pairs)
            pairs = [
                (l, g)
                for l, g in pairs
                if self.clustering.is_enrolled(g)
                or float(np.max(seg_np[:, int(l)]))
                >= self._unknown_emit_min_seg_max
            ]
            used_local = {l for l, _ in pairs}
            if _pairs_before > len(pairs):
                # #region agent log
                import json as _j
                import time as _t

                open(
                    "/home/michel/Documents/Voice/.cursor/debug-21cffc.log", "a"
                ).write(
                    _j.dumps(
                        {
                            "sessionId": "21cffc",
                            "hypothesisId": "H8",
                            "location": "pipeline.py:unknown_filter",
                            "message": "UNKNOWN_PAIR_DROP",
                            "data": {
                                "step": self._step_idx,
                                "before": int(_pairs_before),
                                "after": int(len(pairs)),
                                "min_smx": float(self._unknown_emit_min_seg_max),
                            },
                            "timestamp": int(_t.time() * 1000),
                        }
                    )
                    + "\n"
                )
                # #endregion

        # #region agent log
        if self._step_idx % 2 == 0 or len(pairs) == 0:
            import json as _j
            import time as _t

            _locals = []
            for i in range(self.n_local_speakers):
                _locals.append(
                    {
                        "loc": i,
                        "emb": bool(source_stats[i]["has_embedding"]),
                        "vs": round(float(source_stats[i]["voiced_sec"]), 4),
                        "smx": round(float(np.max(seg_np[:, i])), 4),
                        "act": bool(active_mask[i]),
                    }
                )
            _ho_meta = []
            if self.clustering._anchors:
                for _g in sorted(self.clustering._anchors.keys()):
                    _ls = self.clustering._last_enrolled_step.get(_g, -999)
                    _ho_meta.append(
                        {
                            "g": int(_g),
                            "call_at_last_match": int(_ls),
                            "steps_since": int(self.clustering._call_count - _ls),
                            "prev_loc": self.clustering._last_local_for_enrolled.get(_g),
                        }
                    )
            open(
                "/home/michel/Documents/Voice/.cursor/debug-21cffc.log", "a"
            ).write(
                _j.dumps(
                    {
                        "sessionId": "21cffc",
                        "hypothesisId": "H1",
                        "location": "pipeline.py:step",
                        "message": "STEP_PAIRS",
                        "data": {
                            "step": self._step_idx,
                            "call": self.clustering._call_count,
                            "n_pairs": len(pairs),
                            "pairs": [[int(a), int(b)] for a, b in pairs],
                            "locals": _locals,
                            "holdover": _ho_meta,
                            "hmax_steps": int(self._holdover_max_steps),
                            "holdover_no_emb_cap": int(self._holdover_no_emb_max_steps),
                        },
                        "timestamp": int(_t.time() * 1000),
                    }
                )
                + "\n"
            )
        # #endregion

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
                        # #region agent log
                        if self._step_idx % 4 == 0:
                            import json as _j
                            import time as _t

                            open(
                                "/home/michel/Documents/Voice/.cursor/debug-21cffc.log",
                                "a",
                            ).write(
                                _j.dumps(
                                    {
                                        "sessionId": "21cffc",
                                        "hypothesisId": "H7",
                                        "location": "pipeline.py:xfade",
                                        "message": "XFADE_SKIP_ONSET",
                                        "data": {
                                            "step": self._step_idx,
                                            "g": int(global_idx),
                                            "prev_rms": round(prev_e, 5),
                                            "new_rms": round(new_e, 5),
                                        },
                                        "timestamp": int(_t.time() * 1000),
                                    }
                                )
                                + "\n"
                            )
                        # #endregion
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

        self._step_idx += 1
        infer_ms = (time.perf_counter() - t0) * 1000

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
            unknown_reuse_delta=clust_cfg.get("unknown_reuse_delta"),
            unknown_min_voiced_sec=clust_cfg.get("unknown_min_voiced_sec", 0.18),
        )
        if old_anchors:
            self.clustering.inject_centroids(old_anchors)
