"""
RealtimePipeline — orchestrates DeepFilterNet → PixIT → WeSpeaker → EnrolledClustering.

Current architecture
────────────────────
Identity follows diart's official embedding idea: a mixed waveform plus
per-speaker segmentation weights. PixIT-separated channels are still useful,
but only after identity is decided:

1. overlap-aware mixed-waveform embeddings for both enrollment and live matching
2. separated-source reconstruction / diagnostics
3. never as "instant stable causal speaker streams" for live playback

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
    → source mapping: per-speaker masked latent features + activity from aggregated seg
    → Hamming aggregation of global masked latent features across overlapping chunks
    → decode only the emitted delayed step back to waveform
    → [optional] DeepFilterNet on separated audio for enrolled speakers only

Historical debugging summary
───────────────────────────
These notes are here because we spent a long time chasing symptoms that looked
like "bad models" but were actually integration bugs:

- Infer-ms spikes:
  We previously built variable-length per-source waveform crops for WeSpeaker
  and sometimes retried on larger buffers. That made one step turn into several
  unequal embedding forwards and pushed inference from ~240 ms to ~480+ ms.
  The fix was to stop doing custom long variable crops for clustering and to use
  a fixed recent window with diart's overlap-aware mixed-waveform embedding path.

- Enrollment mismatch:
  Using one brittle frozen vector was not enough. Enrollment is now re-extracted
  through the same overlap-aware embedding path as live audio, and we keep a
  small same-space enrollment profile so first-word variations do not
  immediately become ``Unknown-*``.

- Raw PixIT live audio:
  PixIT is a fixed-number separator on overlapping 5 s windows. A local source
  at time ``t`` is not a stable realtime speaker track by itself. Streaming raw
  current-window separated tails caused weak onsets, leakage, and chopped words.
  Live playback now reconstructs from PixIT's masked latent features instead of
  overlap-adding already-decoded waveforms. Global speaker alignment still
  happens chunk-by-chunk, but decoding happens only after latent aggregation.

- Gate/audio latency mismatch:
  We once gated emitted audio with the 2.0 s activity smoother while the audio
  packets themselves were delayed only 1.0 s. That shifted the mute/unmute
  decision by whole 500 ms steps and created the "good window / bad window"
  effect. The emit-time gate now has its own aggregation path aligned to the
  actual output latency.

- Dashboard audio vs visuals:
  The browser originally scheduled chunks by websocket arrival time only. That
  made the heard audio periodically "disconnect/reconnect" even while backend
  packets and visuals were continuous. The binary audio packet header already
  carries ``step_idx``; the dashboard now uses it for gapless scheduling.

Playback quality and output alignment
────────────────────────────────────
PixIT is re-run every 500 ms on a sliding 5 s window, so the newest 500 ms
slice exists in only one separator window. Emitting that freshest decoded slice
directly makes onsets weak and unstable. The backend therefore emits a
**delayed step** whose masked latent features are already covered by at least
two overlapping PixIT windows, then decodes only that aggregated delayed step.
This mirrors diart's latency-oriented design: real time with a controlled
output delay rather than zero-latency edge audio.

Clustered separated sources are also leakage-pruned before output. PixIT always
produces a fixed number of local sources, and more than one local can contain
the same real speaker. Those secondary locals must be treated as leakage of the
matched enrolled speaker, not as new ``Unknown-*`` speakers.

See also: module docstring in ``enrolled_clustering.py`` and comments in
``config.yaml`` for tunables.
"""
from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.spatial.distance import cdist
import torchaudio.functional as AF
import yaml
from pyannote.core import SlidingWindow, SlidingWindowFeature

from diart.blocks.aggregation import DelayedAggregation
from diart.blocks.embedding import OverlapAwareSpeakerEmbedding
from embedding_runtime import (
    embedding_model_label,
    extract_embedding_vector,
    load_embedding_model,
)
from enrollment_store import embedding_cache_path, reference_paths
from enrolled_clustering import EnrolledSpeakerClustering
from pixit_wrapper import make_pixit_segmentation_model

log = logging.getLogger(__name__)

_RUNTIME_PROFILE_SCHEMA = "nexus.voice.runtime_enrollment_profile.v1"
_RUNTIME_PROFILE_NPY = "embedding_profile.npy"
_RUNTIME_PROFILE_JSON = "embedding_profile.json"


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
      audio gated by an aligned global speaker activity mask, because raw
      separated PixIT locals are not stable causal speaker streams at onsets.

    Design invariants
    -----------------
    If you refactor this file later, these invariants are the important ones to
    preserve:

    - Unknown-speaker clustering stays as close to vanilla diart as practical.
    - Persistent enrolled identity must stay in the same overlap-aware
      embedding space as live clustering. Do not reintroduce a second verifier
      space for enrollment or live matching.
    - Live playback and visualization do not have to use the same aggregation
      latency. UI can be smoother; emitted audio must be aligned to its own
      packet latency.
    - Do not revert to raw current-window PixIT-local playback for enrolled
      speech unless you intentionally accept more onset artifacts.

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

        self._embedding_model_name = str(emb_cfg["model"])
        log.info(
            "Loading speaker embedding model: %s (%s)",
            self._embedding_model_name,
            embedding_model_label(self._embedding_model_name),
        )
        self._emb_model = load_embedding_model(
            self._embedding_model_name,
            self.device,
        )

        # Keep identity close to official diart: one mixed waveform chunk plus
        # per-speaker segmentation weights for both live matching and
        # enrollment rebuilds. Earlier split-space verifier paths caused the
        # enrolled lane to react to separated-source junk that clustering did
        # not consider the same way.
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
        self._adaptive_profile_enabled = bool(
            emb_cfg.get("adaptive_profile_enabled", True)
        )
        self._adaptive_profile_consecutive_steps = max(
            1, int(emb_cfg.get("adaptive_profile_consecutive_steps", 3))
        )
        self._adaptive_profile_max_rows = max(
            self._enroll_profile_top_k,
            int(emb_cfg.get("adaptive_profile_max_rows", 12)),
        )
        self._adaptive_profile_min_activity = float(
            emb_cfg.get("adaptive_profile_min_activity", 0.55)
        )
        self._adaptive_profile_min_voiced_sec = float(
            emb_cfg.get(
                "adaptive_profile_min_voiced_sec",
                max(0.35, self._embed_min_voiced_sec * 3.0),
            )
        )
        self._adaptive_profile_min_tail_rms = float(
            emb_cfg.get("adaptive_profile_min_tail_rms", 0.03)
        )
        self._adaptive_profile_min_similarity = float(
            emb_cfg.get(
                "adaptive_profile_min_similarity",
                max(0.0, 1.0 - float(clust_cfg.get("delta_enrolled", 0.80)) + 0.08),
            )
        )
        self._adaptive_profile_min_row_distance = float(
            emb_cfg.get("adaptive_profile_min_row_distance", 0.05)
        )
        self._adaptive_profile_competitor_margin = float(
            emb_cfg.get("adaptive_profile_competitor_margin", 0.05)
        )
        self._adaptive_profile_cooldown_steps = max(
            0,
            int(
                round(
                    float(emb_cfg.get("adaptive_profile_cooldown_sec", 2.0))
                    / self.step_duration
                )
            ),
        )
        self._enroll_scan_step_sec = float(
            emb_cfg.get("enrollment_scan_step_sec", self.step_duration)
        )
        self._enroll_scan_step_sec = max(
            1.0 / float(self.sample_rate),
            min(self.duration, self._enroll_scan_step_sec),
        )
        self._enroll_min_voiced_sec = float(
            emb_cfg.get(
                "enrollment_min_voiced_sec",
                max(0.40, self._embed_min_voiced_sec * 4.0),
            )
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
        # Mid-utterance rescue is much stricter than onset rescue. Use it only
        # when the enrolled lane was assigned to a very weak local while a much
        # stronger local in the same chunk is almost equally close to the
        # enrolled anchor. This specifically targets PixIT local-source flicker.
        self._flicker_aux_dominance_ratio = 2.0
        self._flicker_aux_dist_margin = 0.05
        self._flicker_aux_min_tail_rms = 0.05
        # When PixIT splits one real enrolled speaker across multiple locals,
        # do not sum them. Instead, compose the delayed packet block-wise from
        # the strongest nearby enrolled-like local to avoid phase doubling.
        self._enrolled_merge_block_sec = 0.04
        self._enrolled_merge_dist_margin = 0.05
        self._enrolled_merge_switch_ratio = 1.15
        self._enrolled_merge_min_block_rms = 0.01

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
            enrolled_continuity_margin=clust_cfg.get(
                "enrolled_continuity_margin", 0.20
            ),
            enrolled_continuity_max_gap=clust_cfg.get(
                "enrolled_continuity_max_gap", 3
            ),
        )
        self._enrollment_profiles_by_name: Dict[str, np.ndarray] = {}
        if enrolled_embeddings:
            # Re-extract through the same overlap-aware identity path as live.
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
        # Onset fallback is only about speaker-map continuity now. A slightly
        # lower activity floor lets a newly recovered enrolled speaker emit its
        # first packet without needing a separate sample-level audio gate.
        self._enrolled_onset_activity_threshold = float(
            acfg.get(
                "enrolled_onset_activity_threshold",
                acfg.get("enrolled_onset_gate_threshold", 0.01),
            )
        )
        # pyannote's SpeechSeparation pipeline has a leakage-removal stage that
        # zeroes separated sources when the aligned speaker diarization is
        # inactive. Keep a causal approximation for unknown-lane output, but do
        # not apply it to enrolled packets after identity is already decided.
        self._separated_leakage_removal = bool(
            acfg.get("separated_leakage_removal", True)
        )
        self._separated_leakage_collar_sec = float(
            acfg.get("separated_leakage_collar_sec", 0.10)
        )
        self._separated_gate_threshold = float(
            acfg.get("separated_gate_threshold", 0.01)
        )
        # Verification gates: suppress enrolled audio when the identity match
        # is weak. This prevents false-positive enrolled audio from leaking
        # through when clustering happens to match a wrong local to the enrolled
        # anchor. Similarity is cosine similarity (1 - cosine_distance).
        self._enrolled_min_similarity = float(
            acfg.get("enrolled_min_similarity", 0.25)
        )
        self._enrolled_min_activity = float(
            acfg.get("enrolled_min_activity", 0.10)
        )
        self._enrolled_onset_min_activity = float(
            acfg.get(
                "enrolled_onset_min_activity",
                min(self._enrolled_min_activity, 0.02),
            )
        )
        self._enrolled_tail_rms_gate = float(
            acfg.get("enrolled_tail_rms_gate", 0.0)
        )
        self._enrolled_mic_gate = float(
            acfg.get("enrolled_mic_gate", 0.0)
        )
        self._enrolled_mic_gate_holdover = max(
            0, int(acfg.get("enrolled_mic_gate_holdover_steps", 1))
        )
        log.info(
            "Separated-audio output latency: %.3fs (%d step%s)",
            self._output_latency_sec,
            self._output_latency_steps,
            "" if self._output_latency_steps == 1 else "s",
        )
        self._latent_aggregation = DelayedAggregation(
            self.step_duration,
            latency=self._output_latency_sec,
            strategy="hamming",
            cropping_mode="loose",
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
        self._latent_buffer: List[SlidingWindowFeature] = []
        self._latent_valid_buffer: List[SlidingWindowFeature] = []
        self._emit_pred_buffer: List[SlidingWindowFeature] = []
        self._emit_meta_buffer: List[tuple[int, List[Dict[str, Any]]]] = []
        self._last_emitted_step_by_global: Dict[int, int] = {}
        self._latent_filters = 0
        self._latent_frames_per_chunk = 0
        self._latent_step_frames = 0
        self._pixit_dtype = torch.float16
        self._adaptive_profile_streaks: Dict[int, int] = {}
        self._adaptive_profile_last_add_step: Dict[int, int] = {}
        self._mic_step_rms_buffer: Dict[int, float] = {}
        if self._step_perf_log and self._step_perf_path is not None:
            log.info(
                "Step perf trace enabled: %s (every_step=%s, spike>=%.1f ms)",
                self._step_perf_path,
                self._step_perf_every_step,
                self._step_perf_spike_threshold_ms,
            )

        # Warmup both models + detect embedding dim
        log.info(
            "Warming up PixIT + %s...",
            embedding_model_label(self._embedding_model_name),
        )
        dummy_fp16 = torch.randn(1, 1, self.chunk_samples, device=self.device, dtype=torch.float16)
        dummy_fp32 = dummy_fp16.float()
        with torch.inference_mode():
            self._pixit_wrapper(dummy_fp16)       # PixIT is fp16
            if self._pixit_wrapper.last_masked_tf_rep is None:
                raise RuntimeError("PixIT wrapper did not expose masked latent features")
            latent_shape = self._pixit_wrapper.last_masked_tf_rep.shape
            self._latent_filters = int(latent_shape[2])
            self._latent_frames_per_chunk = int(latent_shape[3])
            self._latent_step_frames = max(
                1,
                int(round(self._latent_frames_per_chunk * self.step_duration / self.duration)),
            )
            self._pixit_dtype = self._pixit_wrapper.last_masked_tf_rep.dtype
            d = self._emb_model(dummy_fp32)
            self._emb_dim = d.shape[-1]
            dummy_waveform = torch.randn(
                1, self.chunk_samples, 1, device=self.device, dtype=torch.float32
            )
            dummy_seg = torch.rand(
                1, 624, self.n_local_speakers, device=self.device, dtype=torch.float32
            )
            self._oa_emb(dummy_waveform, dummy_seg)
        torch.cuda.synchronize(self.device)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        log.info(
            "Pipeline ready (emb_dim=%d, latent=%d filters x %d frames).",
            self._emb_dim,
            self._latent_filters,
            self._latent_frames_per_chunk,
        )

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
        """Extract an L2-normalized speaker embedding from raw mono audio.

        Used for diagnostics and compatibility fallbacks.

        Parameters
        ----------
        audio : np.ndarray, shape (samples,), float32, 16 kHz

        Returns
        -------
        np.ndarray, shape (emb_dim,), float64, L2-normalized
        """
        return extract_embedding_vector(self._emb_model, audio, self.device)

    def _segmentation_stats(
        self,
        seg_track: np.ndarray,
        window_sec: float,
    ) -> Dict[str, float]:
        """Summarize one segmentation track over the embedding window."""
        seg_arr = np.asarray(seg_track, dtype=np.float32).reshape(-1)
        frame_sec = float(window_sec) / float(max(1, len(seg_arr)))
        voiced_frames = int(np.count_nonzero(seg_arr >= self._embed_seg_threshold))
        return {
            "voiced_sec": voiced_frames * frame_sec,
            "voiced_ratio": voiced_frames / float(max(1, len(seg_arr))),
            "seg_max": float(np.max(seg_arr)) if len(seg_arr) else 0.0,
            "seg_mean": float(np.mean(seg_arr)) if len(seg_arr) else 0.0,
        }

    def _embedding_window(
        self,
        waveform: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop waveform + segmentation to the fixed identity-matching window."""
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

    def _extract_overlap_aware_embeddings(
        self,
        waveform: np.ndarray,
        segmentation: np.ndarray,
    ) -> np.ndarray:
        """Run diart's overlap-aware embedder on one waveform/segmentation pair."""
        with torch.inference_mode():
            embs = self._oa_emb(
                waveform[np.newaxis, :, np.newaxis],
                segmentation[np.newaxis, :, :],
            )
        embs = torch.as_tensor(embs, dtype=torch.float32)
        if embs.ndim == 3 and embs.shape[0] == 1:
            embs = embs[0]
        if embs.ndim == 1:
            embs = embs.unsqueeze(0)
        return embs.detach().cpu().numpy().astype(np.float64, copy=False)

    def _chunk_start_time(self, step_idx: int) -> float:
        """Absolute start time of the chunk processed at ``step_idx``."""
        return step_idx * self.step_duration - (self.duration - self.step_duration)

    def _scan_chunk_starts(self, total_samples: int, hop_samples: int) -> List[int]:
        """Return chunk starts for a sliding scan over a waveform."""
        total_samples = int(total_samples)
        hop_samples = max(1, int(hop_samples))
        if total_samples <= self.chunk_samples:
            return [0]
        starts = list(range(0, total_samples - self.chunk_samples + 1, hop_samples))
        tail_start = total_samples - self.chunk_samples
        if starts[-1] != tail_start:
            starts.append(tail_start)
        return starts

    def _rank_enrollment_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[int]:
        if not candidates:
            return []
        rows = np.stack(
            [cand["embedding"] for cand in candidates],
            axis=0,
        ).astype(np.float64, copy=False)
        dist_matrix = cdist(rows, rows, metric="cosine")
        return sorted(
            range(len(candidates)),
            key=lambda idx: (
                float(dist_matrix[idx].mean()),
                -float(candidates[idx]["stats"]["voiced_sec"]),
                -float(candidates[idx]["stats"]["seg_mean"]),
                -float(candidates[idx]["stats"]["seg_max"]),
                int(candidates[idx]["start"]),
                int(candidates[idx]["local_idx"]),
            ),
        )

    def _scan_reference_enrollment_candidates(
        self,
        audio: np.ndarray,
        ref_name: str,
    ) -> Dict[str, Any]:
        enroll_hop_samples = int(round(self._enroll_scan_step_sec * self.sample_rate))
        chunk_starts = self._scan_chunk_starts(len(audio), enroll_hop_samples)

        strong_candidates: List[Dict[str, Any]] = []
        accepted_candidates: List[Dict[str, Any]] = []
        fallback_candidates: List[Dict[str, Any]] = []

        with torch.inference_mode():
            for start in chunk_starts:
                chunk = audio[start : start + self.chunk_samples]
                if len(chunk) < self.chunk_samples:
                    chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))

                wav_gpu = torch.from_numpy(chunk).to(
                    device=self.device,
                    dtype=torch.float16,
                )
                wav_gpu = wav_gpu.unsqueeze(0).unsqueeze(0)
                seg_t = self._pixit_wrapper(wav_gpu)
                seg_np = seg_t[0].float().cpu().numpy()
                embed_waveform, embed_seg = self._embedding_window(chunk, seg_np)
                window_sec = float(embed_waveform.shape[0]) / float(self.sample_rate)
                oa_embs = self._extract_overlap_aware_embeddings(
                    embed_waveform,
                    embed_seg,
                )

                for spk_idx in range(embed_seg.shape[1]):
                    emb = oa_embs[spk_idx]
                    if np.isnan(emb).any():
                        continue
                    stats = self._segmentation_stats(
                        embed_seg[:, spk_idx],
                        window_sec,
                    )
                    candidate = {
                        "embedding": emb,
                        "stats": stats,
                        "local_idx": int(spk_idx),
                        "ref_name": ref_name,
                        "start": int(start),
                    }
                    fallback_candidates.append(candidate)
                    if stats["voiced_sec"] >= self._embed_min_voiced_sec:
                        accepted_candidates.append(candidate)
                    if stats["voiced_sec"] >= self._enroll_min_voiced_sec:
                        strong_candidates.append(candidate)

        if strong_candidates:
            selected = strong_candidates
            pool_name = "strong"
        elif accepted_candidates:
            selected = accepted_candidates
            pool_name = "accepted"
        else:
            selected = fallback_candidates
            pool_name = "fallback"

        ranked = self._rank_enrollment_candidates(selected)
        representative = selected[ranked[0]] if ranked else None
        return {
            "accepted_candidates": accepted_candidates,
            "fallback_candidates": fallback_candidates,
            "pool_name": pool_name,
            "selected_candidates": selected,
            "representative": representative,
            "strong_candidates": strong_candidates,
        }

    def _aggregate_latent_chunk(
        self,
        step_idx: int,
        chunk_latent: np.ndarray,
    ) -> np.ndarray:
        """Aggregate global masked latent features over overlapping windows."""
        latent = np.asarray(chunk_latent, dtype=np.float32)
        if latent.ndim != 3 or latent.shape[0] != self._latent_frames_per_chunk:
            return np.zeros(
                (
                    self._latent_step_frames,
                    self.clustering.max_speakers,
                    self._latent_filters,
                ),
                dtype=np.float32,
            )

        latent_sw = SlidingWindow(
            start=self._chunk_start_time(step_idx),
            duration=self.duration / float(self._latent_frames_per_chunk),
            step=self.duration / float(self._latent_frames_per_chunk),
        )
        flat = latent.reshape(self._latent_frames_per_chunk, -1)
        valid = (
            np.sqrt(np.mean(latent.astype(np.float64) ** 2, axis=2)) > 1e-8
        ).astype(np.float32)
        valid_flat = np.repeat(valid[:, :, np.newaxis], self._latent_filters, axis=2).reshape(
            self._latent_frames_per_chunk,
            -1,
        )
        self._latent_buffer.append(SlidingWindowFeature(flat, latent_sw))
        self._latent_valid_buffer.append(SlidingWindowFeature(valid_flat, latent_sw))
        agg_latent = self._latent_aggregation(self._latent_buffer)
        agg_valid = self._latent_aggregation(self._latent_valid_buffer)
        if len(self._latent_buffer) == self._latent_aggregation.num_overlapping_windows:
            self._latent_buffer = self._latent_buffer[1:]
            self._latent_valid_buffer = self._latent_valid_buffer[1:]

        data_flat = np.asarray(agg_latent.data, dtype=np.float32)
        valid_mean_flat = np.asarray(agg_valid.data, dtype=np.float32)
        if data_flat.shape[0] < self._latent_step_frames:
            data_flat = np.pad(
                data_flat,
                ((0, self._latent_step_frames - data_flat.shape[0]), (0, 0)),
                mode="constant",
            )
            valid_mean_flat = np.pad(
                valid_mean_flat,
                ((0, self._latent_step_frames - valid_mean_flat.shape[0]), (0, 0)),
                mode="constant",
            )
        elif data_flat.shape[0] > self._latent_step_frames:
            data_flat = data_flat[: self._latent_step_frames]
            valid_mean_flat = valid_mean_flat[: self._latent_step_frames]

        data = data_flat.reshape(
            self._latent_step_frames,
            self.clustering.max_speakers,
            self._latent_filters,
        )
        valid_mean = valid_mean_flat.reshape(
            self._latent_step_frames,
            self.clustering.max_speakers,
            self._latent_filters,
        )
        for global_idx in range(self.clustering.max_speakers):
            if not self.clustering.is_enrolled(int(global_idx)):
                continue
            denom = np.maximum(valid_mean[:, global_idx, :], 1e-6)
            data[:, global_idx, :] = np.divide(
                data[:, global_idx, :],
                denom,
                out=np.zeros_like(data[:, global_idx, :]),
                where=valid_mean[:, global_idx, :] > 1e-6,
            )
        return data

    def _decode_aggregated_latent_step(
        self,
        aggregated_latent: np.ndarray,
    ) -> np.ndarray:
        """Decode one aggregated latent step back to waveform."""
        latent = np.asarray(aggregated_latent, dtype=np.float32)
        if latent.ndim != 3:
            return np.zeros((self.step_samples, self.clustering.max_speakers), dtype=np.float32)

        latent_t = torch.from_numpy(
            np.transpose(latent, (1, 2, 0))
        ).unsqueeze(0).to(device=self.device, dtype=self._pixit_dtype)
        with torch.inference_mode():
            decoded = self._pixit_wrapper.decode_masked_tf_rep(
                latent_t,
                self.step_samples,
            )
        audio = decoded[0].float().cpu().numpy().astype(np.float32, copy=False)
        latent_energy = np.sqrt(
            np.mean(np.asarray(latent, dtype=np.float64) ** 2, axis=(0, 2))
        )
        silent_globals = np.where(latent_energy < 1e-8)[0]
        if silent_globals.size:
            audio[:, silent_globals] = 0.0
        return audio

    def _step_track_mask(
        self,
        step_track: np.ndarray,
        collar_sec: float,
        threshold: float,
    ) -> np.ndarray:
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
        active = track >= float(threshold)
        if collar_sec > 0.0:
            frame_sec = self.step_duration / float(max(1, track.size))
            collar_frames = max(0, int(round(collar_sec / frame_sec)))
            if collar_frames > 0:
                kernel = np.ones(2 * collar_frames + 1, dtype=np.float32)
                padded = np.pad(
                    active.astype(np.float32),
                    (collar_frames, collar_frames),
                    mode="constant",
                )
                active = np.convolve(padded, kernel, mode="valid") > 0
        # Keep the gate binary at sample resolution. Linear interpolation here
        # turns 0/1 frame activity into ramps, which leaks low-level audio at
        # boundaries and is not what pyannote's leakage-removal mask does.
        sample_to_frame = np.minimum(
            track.size - 1,
            (np.arange(self.step_samples, dtype=np.int64) * track.size) // self.step_samples,
        )
        return active.astype(np.float32)[sample_to_frame]

    def _apply_step_track_gate(
        self,
        samples: np.ndarray,
        step_track: np.ndarray,
        collar_sec: float,
        threshold: float,
    ) -> np.ndarray:
        """Apply an aligned binary activity gate to one emitted packet."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        mask = self._step_track_mask(step_track, collar_sec, threshold)
        if mask.shape[0] != audio.shape[0]:
            mask = np.resize(mask, audio.shape[0]).astype(np.float32)
        return (audio * mask).astype(np.float32, copy=False)

    def _candidate_emit_globals(
        self,
        delayed_meta_by_global: Dict[int, Dict[str, Any]],
        current_meta_by_global: Dict[int, Dict[str, Any]],
        emit_step_scores: np.ndarray,
    ) -> set[int]:
        """Pick which globals should emit for the current delayed packet.

        Delayed metadata is the normal source of truth. For enrolled speakers
        only, fall back to the current step's stable label when the delayed
        track is already active but the speaker was assigned one step later.
        That fallback must stay strict: continuity rescues are allowed to keep
        identity stable for assignment, but they must not backfill delayed
        packets as if they were a fresh direct onset match.
        """
        candidate_globals = {int(idx) for idx in delayed_meta_by_global.keys()}
        if emit_step_scores.ndim != 2:
            return candidate_globals

        clustering = getattr(self, "clustering", None)
        strict_similarity_floor = max(
            0.0,
            1.0 - float(getattr(clustering, "delta_enrolled", 1.0)),
        )
        for global_idx, meta in current_meta_by_global.items():
            if global_idx in candidate_globals or not bool(meta.get("is_enrolled", False)):
                continue
            if global_idx >= emit_step_scores.shape[1]:
                continue
            id_sim = meta.get("identity_similarity")
            if id_sim is None or float(id_sim) < strict_similarity_floor:
                continue
            global_track = np.asarray(
                emit_step_scores[:, global_idx], dtype=np.float32
            ).reshape(-1)
            if (
                global_track.size
                and float(np.max(global_track)) >= self._enrolled_onset_activity_threshold
            ):
                candidate_globals.add(int(global_idx))
        return candidate_globals

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

    def _flicker_aux_enrolled_pair(
        self,
        pairs: List[tuple[int, int]],
        active_indices: List[int],
        step_tail_rms_np: np.ndarray,
        embeddings: torch.Tensor,
    ) -> tuple[List[tuple[int, int]], int]:
        """Rescue enrolled audio when PixIT flips to a weak local mid-utterance.

        Unlike onset rescue, this is allowed after the utterance has been going
        on for a while, but it stays deliberately strict: the replacement local
        must be much stronger and still very close to the enrolled anchor.
        """
        if len(self.clustering._anchors) != 1 or not active_indices:
            return pairs, 0

        enrolled_pairs = [
            (int(local), int(global_idx))
            for local, global_idx in pairs
            if self.clustering.is_enrolled(int(global_idx))
        ]
        if not enrolled_pairs:
            return pairs, 0

        assigned_local, anchor_idx = enrolled_pairs[0]
        assigned_emb = (
            embeddings[assigned_local].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        if np.isnan(assigned_emb).any():
            return pairs, 1

        assigned_tail = float(step_tail_rms_np[assigned_local])
        assigned_dist = float(
            self.clustering.enrolled_distance(anchor_idx, assigned_emb)
        )
        best_local = assigned_local
        best_tail = assigned_tail
        best_dist = assigned_dist

        for local_idx in (int(idx) for idx in active_indices):
            if local_idx == assigned_local:
                continue
            emb_row = (
                embeddings[local_idx].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            if np.isnan(emb_row).any():
                continue

            tail_rms = float(step_tail_rms_np[local_idx])
            if tail_rms < self._flicker_aux_min_tail_rms:
                continue
            if tail_rms < max(
                self._flicker_aux_min_tail_rms,
                assigned_tail * self._flicker_aux_dominance_ratio,
            ):
                continue

            dist = float(self.clustering.enrolled_distance(anchor_idx, emb_row))
            if dist > (self.clustering.delta_enrolled + self._flicker_aux_dist_margin):
                continue
            if dist > (assigned_dist + self._flicker_aux_dist_margin):
                continue

            if tail_rms > best_tail:
                best_local = local_idx
                best_tail = tail_rms
                best_dist = dist

        if best_local == assigned_local:
            return pairs, 1

        out_pairs = [
            (l, g)
            for l, g in pairs
            if int(g) != anchor_idx and int(l) != best_local
        ]
        out_pairs.append((best_local, anchor_idx))
        return out_pairs, 1

    def _compose_enrolled_chunk_latent(
        self,
        assigned_local: int,
        anchor_idx: int,
        active_indices: List[int],
        embeddings: torch.Tensor,
        source_stats: List[Dict[str, float]],
        segmentation: np.ndarray,
        masked_tf_np: np.ndarray,
    ) -> np.ndarray:
        """Compose enrolled chunk latent features from very-near locals.

        The root problem is not just thresholding: blind separation can split
        one real speaker across more than one local channel in the same chunk.
        If we keep only the assigned local, speech can vanish when PixIT flips
        energy into a nearby unknown local. If we sum locals after decode, the
        waveforms interfere. Instead, pick a block-wise latent winner across
        locals that are still close enough to the same enrolled anchor.
        """
        assigned_local = int(assigned_local)
        base = np.asarray(masked_tf_np[assigned_local], dtype=np.float32)
        if not active_indices:
            return np.transpose(base, (1, 0))

        assigned_emb = (
            embeddings[assigned_local].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        if np.isnan(assigned_emb).any():
            return np.transpose(base, (1, 0))

        assigned_dist = float(self.clustering.enrolled_distance(anchor_idx, assigned_emb))
        max_dist = max(
            self.clustering.delta_enrolled + self._enrolled_merge_dist_margin,
            assigned_dist + self._enrolled_merge_dist_margin,
        )
        eligible: List[int] = [assigned_local]
        dist_by_local: Dict[int, float] = {assigned_local: assigned_dist}
        for local_idx in (int(idx) for idx in active_indices):
            if local_idx == assigned_local:
                continue
            if not bool(source_stats[local_idx]["has_embedding"]):
                continue
            emb_row = (
                embeddings[local_idx].detach().cpu().numpy().astype(np.float64, copy=False)
            )
            if np.isnan(emb_row).any():
                continue
            dist = float(self.clustering.enrolled_distance(anchor_idx, emb_row))
            if dist > max_dist:
                continue
            eligible.append(local_idx)
            dist_by_local[local_idx] = dist

        if len(eligible) == 1:
            return np.transpose(base, (1, 0))

        composed = base.copy()
        prev_local = assigned_local
        block_samples = max(
            1,
            int(round(self._enrolled_merge_block_sec * float(self._latent_frames_per_chunk) / float(self.duration))),
        )
        seg_frames = max(1, int(segmentation.shape[0]))
        latent_frames = max(1, int(base.shape[-1]))

        for block_start in range(0, latent_frames, block_samples):
            block_end = min(latent_frames, block_start + block_samples)
            seg_start = min(
                seg_frames - 1,
                int(np.floor(block_start * seg_frames / float(latent_frames))),
            )
            seg_end = max(
                seg_start + 1,
                min(
                    seg_frames,
                    int(np.ceil(block_end * seg_frames / float(latent_frames))),
                ),
            )

            scores: Dict[int, float] = {}
            best_local = prev_local
            best_score = -1.0
            for local_idx in eligible:
                block = np.asarray(
                    masked_tf_np[local_idx, :, block_start:block_end],
                    dtype=np.float32,
                )
                if block.size == 0:
                    continue
                block_energy = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
                if block_energy < self._enrolled_merge_min_block_rms:
                    continue
                seg_mean = float(
                    np.mean(segmentation[seg_start:seg_end, local_idx])
                )
                # Small distance bias keeps us from grabbing merely loud nearby
                # contamination when two locals are not equally Michel-like.
                score = (block_energy * max(seg_mean, 1e-3)) / (
                    1.0 + dist_by_local[local_idx]
                )
                scores[local_idx] = score
                if score > best_score:
                    best_local = local_idx
                    best_score = score

            prev_score = scores.get(prev_local, 0.0)
            if (
                best_local != prev_local
                and prev_score > 0.0
                and best_score < (prev_score * self._enrolled_merge_switch_ratio)
            ):
                best_local = prev_local

            composed[:, block_start:block_end] = masked_tf_np[
                best_local, :, block_start:block_end
            ]
            prev_local = best_local

        return np.transpose(composed.astype(np.float32, copy=False), (1, 0))

    def _prune_profile_rows(
        self,
        rows: np.ndarray,
    ) -> np.ndarray:
        """Keep the profile bank compact and diverse."""
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        rows = rows / np.maximum(norms, 1e-12)
        while rows.shape[0] > self._adaptive_profile_max_rows:
            dist = cdist(rows, rows, metric="cosine")
            np.fill_diagonal(dist, np.inf)
            drop_idx = int(np.argmin(np.min(dist, axis=1)))
            rows = np.delete(rows, drop_idx, axis=0)
        return rows.astype(np.float64, copy=False)

    def _maybe_update_adaptive_profiles(
        self,
        emit_pairs: List[tuple[int, int, float, bool]],
        active_indices: List[int],
        embeddings: torch.Tensor,
        source_stats: List[Dict[str, float]],
    ) -> int:
        """Promote only repeated clear enrolled matches into the live profile bank."""
        if not self._adaptive_profile_enabled or not emit_pairs:
            return 0

        updates = 0
        active_list = [int(idx) for idx in active_indices]
        for local_idx, global_idx, activity, is_enrolled in emit_pairs:
            global_idx = int(global_idx)
            local_idx = int(local_idx)
            if not is_enrolled:
                continue
            emb_row = embeddings[local_idx].detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            if np.isnan(emb_row).any():
                self._adaptive_profile_streaks[global_idx] = 0
                continue

            label = self.clustering.get_label(global_idx)
            anchor = self.clustering._anchors.get(global_idx)
            if anchor is None:
                self._adaptive_profile_streaks[global_idx] = 0
                continue

            id_sim = float(np.clip(np.dot(emb_row, anchor), -1.0, 1.0))
            own_dist = float(self.clustering.enrolled_distance(global_idx, emb_row))
            own_tail = float(source_stats[local_idx].get("tail_rms", 0.0))
            qualifies = (
                bool(source_stats[local_idx].get("has_embedding", False))
                and float(activity) >= self._adaptive_profile_min_activity
                and float(source_stats[local_idx].get("voiced_sec", 0.0))
                >= self._adaptive_profile_min_voiced_sec
                and own_tail >= self._adaptive_profile_min_tail_rms
                and id_sim >= self._adaptive_profile_min_similarity
                and own_dist <= float(self.clustering.delta_enrolled)
            )
            if qualifies:
                for other_idx in active_list:
                    if other_idx == local_idx:
                        continue
                    if not bool(source_stats[other_idx].get("has_embedding", False)):
                        continue
                    other_emb = embeddings[other_idx].detach().cpu().numpy().astype(
                        np.float64, copy=False
                    )
                    if np.isnan(other_emb).any():
                        continue
                    other_tail = float(source_stats[other_idx].get("tail_rms", 0.0))
                    if other_tail <= 0.0:
                        continue
                    other_dist = float(
                        self.clustering.enrolled_distance(global_idx, other_emb)
                    )
                    if (
                        other_dist <= (own_dist + self._adaptive_profile_competitor_margin)
                        and other_tail >= max(self._adaptive_profile_min_tail_rms, own_tail * 0.75)
                    ):
                        qualifies = False
                        break

            if not qualifies:
                self._adaptive_profile_streaks[global_idx] = 0
                continue

            streak = self._adaptive_profile_streaks.get(global_idx, 0) + 1
            self._adaptive_profile_streaks[global_idx] = streak
            if streak < self._adaptive_profile_consecutive_steps:
                continue

            last_add = self._adaptive_profile_last_add_step.get(global_idx, -10_000)
            if (self._step_idx - last_add) < self._adaptive_profile_cooldown_steps:
                continue

            current_rows = self._enrollment_profiles_by_name.get(label)
            if current_rows is None or np.asarray(current_rows).size == 0:
                current_rows = self.clustering._profiles.get(
                    global_idx, anchor.reshape(1, -1)
                )
            current_rows = np.asarray(current_rows, dtype=np.float64).reshape(
                -1, emb_row.shape[0]
            )
            min_row_dist = float(
                cdist(emb_row.reshape(1, -1), current_rows, metric="cosine").min()
            )
            if min_row_dist <= self._adaptive_profile_min_row_distance:
                continue

            updated_rows = np.vstack([current_rows, emb_row.reshape(1, -1)])
            updated_rows = self._prune_profile_rows(updated_rows)
            self._enrollment_profiles_by_name[label] = updated_rows
            self.clustering.inject_profiles({label: updated_rows})
            self._adaptive_profile_last_add_step[global_idx] = self._step_idx
            updates += 1

        return updates

    @staticmethod
    def _runtime_profile_cache_paths(speaker_dir: Path) -> tuple[Path, Path]:
        return (
            speaker_dir / _RUNTIME_PROFILE_NPY,
            speaker_dir / _RUNTIME_PROFILE_JSON,
        )

    @staticmethod
    def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                np.save(tmp, value)
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _atomic_write_json(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                delete=False,
            ) as tmp:
                json.dump(value, tmp, indent=2, sort_keys=True)
                tmp.write("\n")
                tmp_path = Path(tmp.name)
            tmp_path.replace(path)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _normalize_profile_rows(rows: np.ndarray) -> Optional[np.ndarray]:
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(1, -1)
        if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
            return None
        finite = np.all(np.isfinite(rows), axis=1)
        norms = np.linalg.norm(rows, axis=1)
        keep = finite & (norms > 0)
        if not np.any(keep):
            return None
        rows = rows[keep]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return (rows / np.maximum(norms, 1e-12)).astype(np.float64, copy=False)

    @staticmethod
    def _reference_fingerprints(ref_paths: List[Path]) -> List[dict]:
        fingerprints: List[dict] = []
        for path in ref_paths:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            stat = path.stat()
            fingerprints.append(
                {
                    "name": path.name,
                    "size_bytes": int(stat.st_size),
                    "sha256": digest.hexdigest(),
                }
            )
        return fingerprints

    def _runtime_profile_identity(self, config: dict) -> dict:
        pixit_cfg = config.get("pixit", {})
        denoiser_cfg = config.get("denoiser", {})
        return {
            "sample_rate": int(self.sample_rate),
            "pixit_model": str(pixit_cfg.get("model", "")),
            "pixit_duration": float(self.duration),
            "pixit_step": float(self.step_duration),
            "pixit_max_speakers": int(self.n_local_speakers),
            "embedding_model": self._embedding_model_name,
            "source_seg_threshold": float(self._embed_seg_threshold),
            "source_min_voiced_sec": float(self._embed_min_voiced_sec),
            "source_recent_sec": float(self._embed_recent_sec),
            "enrollment_profile_top_k": int(self._enroll_profile_top_k),
            "enrollment_scan_step_sec": float(self._enroll_scan_step_sec),
            "enrollment_min_voiced_sec": float(self._enroll_min_voiced_sec),
            "gamma": float(self._embed_gamma),
            "beta": float(self._embed_beta),
            "denoiser_enabled": bool(self._denoiser_enabled),
            "denoiser_model": (
                str(denoiser_cfg.get("model", "DeepFilterNet3"))
                if self._denoiser_enabled
                else None
            ),
        }

    def _load_runtime_enrollment_profile(
        self,
        name: str,
        speaker_dir: Path,
        ref_paths: List[Path],
        config: dict,
        fallback_embedding: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        profile_path, metadata_path = self._runtime_profile_cache_paths(speaker_dir)
        if not profile_path.exists() or not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                return None
            if metadata.get("schema") != _RUNTIME_PROFILE_SCHEMA:
                return None
            if metadata.get("identity") != self._runtime_profile_identity(config):
                return None
            if (
                ref_paths
                and metadata.get("references") != self._reference_fingerprints(ref_paths)
            ):
                return None
            profile_rows = self._normalize_profile_rows(np.load(profile_path))
            if profile_rows is None:
                return None
            fallback = np.asarray(fallback_embedding, dtype=np.float64).reshape(-1)
            if fallback.size and profile_rows.shape[1] != fallback.size:
                return None
            anchor_index = int(metadata.get("anchor_index", 0))
            if anchor_index < 0 or anchor_index >= profile_rows.shape[0]:
                anchor_index = 0
            emb = profile_rows[anchor_index].copy()
            log.info(
                "Loaded cached overlap-aware enrollment profile for '%s' (rows=%d dim=%d)",
                name,
                profile_rows.shape[0],
                profile_rows.shape[1],
            )
            return emb, profile_rows
        except Exception as exc:
            log.warning(
                "Failed to load cached overlap-aware enrollment profile for '%s': %s",
                name,
                exc,
            )
            return None

    def _save_runtime_enrollment_profile(
        self,
        name: str,
        speaker_dir: Path,
        profile_rows: np.ndarray,
        anchor_index: int,
        config: dict,
        ref_paths: List[Path],
    ) -> None:
        try:
            normalized_rows = self._normalize_profile_rows(profile_rows)
            if normalized_rows is None:
                return
            anchor_index = max(0, min(int(anchor_index), normalized_rows.shape[0] - 1))
            profile_path, metadata_path = self._runtime_profile_cache_paths(speaker_dir)
            self._atomic_save_npy(profile_path, normalized_rows)
            self._atomic_save_npy(
                embedding_cache_path(speaker_dir),
                normalized_rows[anchor_index].copy(),
            )
            metadata = {
                "schema": _RUNTIME_PROFILE_SCHEMA,
                "speaker": name,
                "identity": self._runtime_profile_identity(config),
                "references": self._reference_fingerprints(ref_paths),
                "anchor_index": anchor_index,
                "rows": int(normalized_rows.shape[0]),
                "dim": int(normalized_rows.shape[1]),
                "created_at_epoch": time.time(),
            }
            self._atomic_write_json(metadata_path, metadata)
        except Exception as exc:
            log.warning(
                "Failed to save overlap-aware enrollment profile cache for '%s': %s",
                name,
                exc,
            )

    def _reextract_enrollments(
        self,
        enrolled_embeddings: Dict[str, np.ndarray],
        config: dict,
    ) -> Dict[str, np.ndarray]:
        """Rebuild enrolled anchors through the same identity path as live audio.

        Reference audio still runs through the frontend stack (DeepFilterNet if
        enabled, then PixIT segmentation), but enrollment vectors now come from
        the same overlap-aware mixed-waveform embedder used at runtime. That
        keeps persistent-ID matching in one embedding space instead of mixing
        weighted-chunk embeddings with separated-source verifier embeddings.
        """
        import soundfile as sf
        from scipy.spatial.distance import cdist

        enroll_dir = Path(__file__).parent / config["enrollment"]["dir"]
        reextracted = {}
        self._enrollment_profiles_by_name = {}
        for name in enrolled_embeddings:
            speaker_dir = enroll_dir / name
            ref_paths = reference_paths(speaker_dir)
            cached_profile = self._load_runtime_enrollment_profile(
                name,
                speaker_dir,
                ref_paths,
                config,
                enrolled_embeddings[name],
            )
            if cached_profile is not None:
                emb, profile_rows = cached_profile
                reextracted[name] = emb
                self._enrollment_profiles_by_name[name] = profile_rows
                continue
            if not ref_paths:
                log.warning("No reference*.wav for '%s', using stored embedding", name)
                reextracted[name] = enrolled_embeddings[name]
                self._enrollment_profiles_by_name[name] = np.asarray(
                    enrolled_embeddings[name], dtype=np.float64
                ).reshape(1, -1)
                continue

            mandatory_candidates: List[Dict[str, Any]] = []
            all_candidates: List[Dict[str, Any]] = []
            weak_reference_count = 0
            for wav_path in ref_paths:
                part, sr = sf.read(str(wav_path), dtype="float32")
                if part.ndim > 1:
                    part = part[:, 0]
                if sr != self.sample_rate:
                    part_t = torch.from_numpy(part).unsqueeze(0)
                    part_t = AF.resample(part_t, sr, self.sample_rate)
                    part = part_t.squeeze(0).numpy()
                log.info(
                    "  '%s' loaded %s (%.1fs)",
                    name,
                    wav_path.name,
                    len(part) / self.sample_rate,
                )

                # DeepFilterNet: process in 500ms blocks (same as live path)
                if self._denoiser_enabled and self._df_model is not None:
                    block_size = int(0.5 * self.sample_rate)
                    denoised = []
                    for start in range(0, len(part), block_size):
                        block = part[start : start + block_size]
                        if len(block) < block_size:
                            block = np.pad(block, (0, block_size - len(block)))
                        block = self.denoise_block(block)
                        denoised.append(block)
                    part = np.concatenate(denoised)[: len(part)]

                ref_scan = self._scan_reference_enrollment_candidates(
                    part,
                    wav_path.name,
                )
                ref_candidates = ref_scan["selected_candidates"]
                ref_best = ref_scan["representative"]
                if ref_best is None:
                    log.warning(
                        "  '%s' had no usable overlap-aware enrollment candidates in %s",
                        name,
                        wav_path.name,
                    )
                    continue

                if ref_scan["pool_name"] != "strong":
                    weak_reference_count += 1
                    log.warning(
                        "  '%s' used %s enrollment windows from %s "
                        "(best voiced=%0.2fs seg_mean=%0.3f seg_max=%0.3f)",
                        name,
                        ref_scan["pool_name"],
                        wav_path.name,
                        float(ref_best["stats"]["voiced_sec"]),
                        float(ref_best["stats"]["seg_mean"]),
                        float(ref_best["stats"]["seg_max"]),
                    )

                mandatory_candidates.append(ref_best)
                all_candidates.extend(ref_candidates)

            if not mandatory_candidates or not all_candidates:
                log.warning(
                    "  '%s' had no usable overlap-aware enrollment candidates; using stored embedding",
                    name,
                )
                emb = np.asarray(enrolled_embeddings[name], dtype=np.float64).reshape(-1)
                self._enrollment_profiles_by_name[name] = emb.reshape(1, -1)
                reextracted[name] = emb
                continue

            keep_target = max(self._enroll_profile_top_k, len(mandatory_candidates))
            kept_candidates: List[Dict[str, Any]] = []
            kept_keys: set[tuple[str, int, int]] = set()

            for candidate in mandatory_candidates:
                key = (
                    str(candidate["ref_name"]),
                    int(candidate["start"]),
                    int(candidate["local_idx"]),
                )
                if key in kept_keys:
                    continue
                kept_keys.add(key)
                kept_candidates.append(candidate)

            for idx in self._rank_enrollment_candidates(all_candidates):
                candidate = all_candidates[idx]
                key = (
                    str(candidate["ref_name"]),
                    int(candidate["start"]),
                    int(candidate["local_idx"]),
                )
                if key in kept_keys:
                    continue
                kept_keys.add(key)
                kept_candidates.append(candidate)
                if len(kept_candidates) >= keep_target:
                    break

            anchor_order = self._rank_enrollment_candidates(kept_candidates)
            profile_rows = np.stack(
                [cand["embedding"] for cand in kept_candidates],
                axis=0,
            ).astype(np.float64, copy=False)
            # Keep the frozen anchor as the medoid candidate among the retained
            # profile rows. This avoids a stale single-vector fallback while
            # still preserving one representative row per reference clip.
            emb = profile_rows[anchor_order[0]].copy()
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                emb = emb / emb_norm
            emb = emb.astype(np.float64, copy=False)

            best = kept_candidates[anchor_order[0]]
            log.info(
                "  '%s' OA anchor ref=%s chunk=%0.2fs local=%d voiced=%0.2fs "
                "seg_mean=%0.3f seg_max=%0.3f refs=%d kept_rows=%d candidates=%d weak_refs=%d",
                name,
                str(best["ref_name"]),
                float(best["start"]) / float(self.sample_rate),
                int(best["local_idx"]),
                float(best["stats"]["voiced_sec"]),
                float(best["stats"]["seg_mean"]),
                float(best["stats"]["seg_max"]),
                len(ref_paths),
                len(kept_candidates),
                len(all_candidates),
                weak_reference_count,
            )

            self._enrollment_profiles_by_name[name] = profile_rows.astype(
                np.float64, copy=False
            )
            reextracted[name] = emb
            self._save_runtime_enrollment_profile(
                name,
                speaker_dir,
                profile_rows,
                int(anchor_order[0]),
                config,
                ref_paths,
            )
            log.info(
                "Re-extracted '%s' through %sPixIT-seg→overlap-aware-%s (dim=%d)",
                name,
                "DF→" if self._denoiser_enabled else "",
                embedding_model_label(self._embedding_model_name),
                len(emb),
            )
        return reextracted

    def step(self, waveform: np.ndarray) -> StepResult:
        """Run one pipeline step on a 5 s audio chunk.

        Output audio path: local masked latent features are first mapped to
        global speakers for the current chunk, then those chunk-level global
        latents are Hamming-aggregated across the recent rolling buffer. The
        emitted audio is decoded only after that aggregation, which is more
        stable than stitching raw decoded tails from one PixIT window at a
        time.

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

        # Raw-mic step RMS: the last step_samples of waveform is the newest
        # 500 ms of mic audio, which is exactly what will be emitted one step
        # later (output_latency_steps=2 → emitting step_idx-1 whose newest
        # 500 ms corresponds to waveform[-step_samples:] stored at step_idx-1).
        # Store keyed by current step_idx so we can look it up at emit time.
        _mic_step_rms = float(
            np.sqrt(np.mean(waveform[-self.step_samples:].astype(np.float64) ** 2))
        )
        self._mic_step_rms_buffer[self._step_idx] = _mic_step_rms
        _mic_prune_before = self._step_idx - self._output_latency_steps - 4
        for _k in [k for k in self._mic_step_rms_buffer if k < _mic_prune_before]:
            del self._mic_step_rms_buffer[_k]

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
        masked_tf_rep = self._pixit_wrapper.last_masked_tf_rep
        if masked_tf_rep is None:
            raise RuntimeError("PixIT wrapper did not provide masked latent features")

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
        embed_window_sec = float(embed_waveform.shape[0]) / float(self.sample_rate)

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
                stats = self._segmentation_stats(
                    embed_seg[:, spk_idx],
                    embed_window_sec,
                )
                stats["tail_rms"] = float(step_tail_rms_np[spk_idx])
                source_stats[spk_idx].update(stats)
                if not active_mask[spk_idx]:
                    continue
                if stats["voiced_sec"] < self._embed_min_voiced_sec:
                    embed_recent_failures += 1
                    continue
                active_indices.append(spk_idx)

        t_prep = tc()

        embed_batches = (
            [{"samples": int(embed_waveform.shape[0]), "count": int(len(active_indices))}]
            if active_indices
            else []
        )

        if active_indices:
            embs = self._extract_overlap_aware_embeddings(
                embed_waveform,
                embed_seg,
            )
            for spk_idx in active_indices:
                embeddings[spk_idx] = torch.from_numpy(embs[spk_idx]).float()
                source_stats[spk_idx]["has_embedding"] = True

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
        # ── 4. Map source latents to global speakers (single window) ──
        # One PixIT forward per step; we keep the masked latent features for
        # each local source, align them to global speakers, aggregate them over
        # overlapping windows, and decode only the emitted delayed step.

        results = []
        pairs: List[tuple[int, int]] = []
        used_local: set[int] = set()
        speaker_map_pairs = 0
        anchor_checks = 0
        remap_count = 0
        dedup_drop_count = 0
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
        pairs, flicker_checks = self._flicker_aux_enrolled_pair(
            pairs,
            active_indices,
            step_tail_rms_np,
            embeddings,
        )
        anchor_checks += flicker_checks
        remap_count = int(pairs != pairs_before_aux)

        pairs_after_reconcile = int(len(pairs))
        unknown_pairs_dropped = 0
        masked_tf_np: Optional[np.ndarray] = None

        t5 = tc()
        chunk_global_latent = np.zeros(
            (
                self._latent_frames_per_chunk,
                self.clustering.max_speakers,
                self._latent_filters,
            ),
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
                    emb_row = embeddings[local_idx].numpy().astype(np.float64, copy=False)
                    if not np.isnan(emb_row).any():
                        nearest_enrolled = min(
                            float(
                                self.clustering.enrolled_distance(
                                    g_idx,
                                    emb_row,
                                )
                            )
                            for g_idx in enrolled_pair_globals
                        )
                        if nearest_enrolled <= (self.clustering.delta_enrolled + 0.03):
                            unknown_pairs_dropped += 1
                            continue
                if masked_tf_np is None:
                    masked_tf_np = (
                        masked_tf_rep[0].detach().float().cpu().numpy().astype(
                            np.float32,
                            copy=False,
                        )
                    )
                latent_chunk = np.transpose(
                    np.asarray(masked_tf_np[local_idx], dtype=np.float32),
                    (1, 0),
                )
                if is_enrolled:
                    latent_chunk = self._compose_enrolled_chunk_latent(
                        local_idx,
                        global_idx,
                        active_indices,
                        embeddings,
                        source_stats,
                        seg_np,
                        masked_tf_np,
                    )
                chunk_global_latent[:, global_idx, :] = latent_chunk
                step_track = seg_np[-local_step_frames:, local_idx]
                emit_pairs.append((local_idx, global_idx, activity, is_enrolled))

        delayed_latent = self._aggregate_latent_chunk(self._step_idx, chunk_global_latent)
        delayed_audio = self._decode_aggregated_latent_step(delayed_latent)
        current_emit_meta: List[Dict[str, Any]] = []
        for local_idx, global_idx, activity, is_enrolled in emit_pairs:
            label = self.clustering.get_label(global_idx)
            id_sim: Optional[float] = None
            if global_idx in self.clustering._anchors:
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
                    "tail_rms": float(step_tail_rms_np[local_idx]),
                }
            )

        self._emit_meta_buffer.append((self._step_idx, current_emit_meta))
        if len(self._emit_meta_buffer) > self._output_latency_steps:
            self._emit_meta_buffer = self._emit_meta_buffer[-self._output_latency_steps :]
        emit_step_idx = self._step_idx - self._output_latency_steps + 1
        delayed_emit_meta: List[Dict[str, Any]] = []
        for hist_step_idx, hist_meta in reversed(self._emit_meta_buffer):
            if hist_step_idx == emit_step_idx:
                delayed_emit_meta = hist_meta
                break
        delayed_meta_by_global: Dict[int, Dict[str, Any]] = {
            int(meta["global_idx"]): meta for meta in delayed_emit_meta
        }
        current_meta_by_global: Dict[int, Dict[str, Any]] = {
            int(meta["global_idx"]): meta for meta in current_emit_meta
        }
        candidate_globals = self._candidate_emit_globals(
            delayed_meta_by_global,
            current_meta_by_global,
            emit_step_scores,
        )
        emit_enrolled_identity_rejects = 0
        emit_enrolled_activity_rejects = 0
        emit_silent_packets = 0
        for global_idx in sorted(candidate_globals):
            meta = delayed_meta_by_global.get(global_idx) or current_meta_by_global.get(
                global_idx
            )
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
            is_onset_fallback = bool(
                is_enrolled
                and global_idx not in delayed_meta_by_global
                and global_idx in current_meta_by_global
            )
            last_emit_step = self._last_emitted_step_by_global.get(int(global_idx), -10_000)
            is_fresh_emit = bool(
                is_enrolled
                and (int(emit_step_idx) - int(last_emit_step)) > 1
            )
            is_onset_packet = bool(is_onset_fallback or is_fresh_emit)
            activity_floor = (
                self._enrolled_onset_min_activity
                if is_onset_packet
                else self._enrolled_min_activity
            )
            separated_gate_track = global_track

            audio = delayed_audio[:, global_idx].astype(np.float32, copy=False)
            # Raw-mic gate: if the microphone was silent at the emitted step
            # the enrolled speaker cannot have been talking. This cuts the
            # PixIT-window tail without relying on PixIT source energy, which
            # stays elevated (~0.10-0.15) at the ambient noise floor even after
            # speech ends and therefore cannot discriminate silence reliably.
            if is_enrolled and self._enrolled_mic_gate > 0.0:
                emit_mic_rms = self._mic_step_rms_buffer.get(emit_step_idx, 1.0)
                if emit_mic_rms < self._enrolled_mic_gate:
                    # Backward holdover: if speech was detected within the last
                    # holdover_steps steps, pass this step too. This prevents
                    # the last syllable of a phrase from being clipped when the
                    # mic RMS drops just below threshold at the trailing edge.
                    hold_ok = any(
                        self._mic_step_rms_buffer.get(emit_step_idx - j, 0.0)
                        >= self._enrolled_mic_gate
                        for j in range(1, self._enrolled_mic_gate_holdover + 1)
                    )
                    if not hold_ok:
                        emit_enrolled_activity_rejects += 1
                        continue
            # Verification gate: suppress enrolled audio when identity is weak
            if is_enrolled and id_sim is not None:
                if id_sim < self._enrolled_min_similarity:
                    emit_enrolled_identity_rejects += 1
                    continue
                if activity < activity_floor:
                    emit_enrolled_activity_rejects += 1
                    continue
                # Tail-RMS gate: the separated-source RMS for the emitted step
                # drops immediately when speech ends, while PixIT activity lags
                # ~4 steps due to the 5s context window. Gate suppresses emission
                # when the source was quiet at the emitted step.
                if self._enrolled_tail_rms_gate > 0.0:
                    tail_rms = float(meta.get("tail_rms", 1.0)) if meta is not None else 1.0
                    if tail_rms < self._enrolled_tail_rms_gate:
                        emit_enrolled_activity_rejects += 1
                        continue
            apply_separated_gate = self._separated_leakage_removal and not is_enrolled
            if apply_separated_gate:
                separated_gate_threshold = self._separated_gate_threshold
                separated_gate_collar_sec = self._separated_leakage_collar_sec
                audio = self._apply_step_track_gate(
                    audio,
                    separated_gate_track,
                    separated_gate_collar_sec,
                    separated_gate_threshold,
                )
            if is_enrolled and self._df_output_enrolled:
                audio = self._enhance_enrolled_audio(audio, global_idx)
            audio = np.clip(audio, -1.0, 1.0)
            packet_rms = float(
                np.sqrt(np.mean(audio.astype(np.float64) ** 2))
            ) if audio.size else 0.0
            if packet_rms < 1e-4:
                emit_silent_packets += 1
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
            self._last_emitted_step_by_global[int(global_idx)] = int(emit_step_idx)

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
                        # Retained for debug-log compatibility. There is no
                        # separate verifier path anymore; identity uses the
                        # same overlap-aware embedding end-to-end.
                        "has_verify_embedding": bool(source_stats[idx]["has_embedding"]),
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
                    "unknown_pairs_dropped": int(unknown_pairs_dropped),
                    "emit_enrolled_identity_rejects": int(emit_enrolled_identity_rejects),
                    "emit_enrolled_activity_rejects": int(emit_enrolled_activity_rejects),
                    "emit_silent_packets": int(emit_silent_packets),
                    "final_pairs": int(len(pairs)),
                    "anchor_count": int(len(self.clustering._anchors)),
                    "anchor_checks": int(anchor_checks),
                    "remap_count": int(remap_count),
                    "dedup_drop_count": int(dedup_drop_count),
                    "enrolled_distances": {
                        self.clustering.get_label(int(k)): v
                        for k, v in self.clustering._last_enrolled_distances.items()
                    },
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
                "id_reject=%d act_reject=%d silent=%d pairs=%d",
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
                emit_enrolled_identity_rejects,
                emit_enrolled_activity_rejects,
                emit_silent_packets,
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
        self._latent_buffer.clear()
        self._latent_valid_buffer.clear()
        self._emit_meta_buffer.clear()
        self._pred_seg_buffer.clear()
        self._emit_pred_buffer.clear()
        self._last_emitted_step_by_global.clear()
        self._df_output_states.clear()
        self._mic_step_rms_buffer.clear()
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
            enrolled_continuity_margin=clust_cfg.get(
                "enrolled_continuity_margin", 0.20
            ),
            enrolled_continuity_max_gap=clust_cfg.get(
                "enrolled_continuity_max_gap", 3
            ),
        )
        if old_anchors:
            self.clustering.inject_centroids(old_anchors)
            self.clustering.inject_profiles(self._enrollment_profiles_by_name)
