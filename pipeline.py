"""
RealtimePipeline — orchestrates DeepFilterNet → PixIT → WeSpeaker → EnrolledClustering.

Uses diart's building blocks (SpeakerSegmentation, OverlapAwareSpeakerEmbedding,
OnlineSpeakerClustering) but NOT diart's SpeakerDiarization pipeline or
StreamingInference — those don't support PixIT's dual output (diarization +
separated sources).

Data flow per step
──────────────────
  waveform np.ndarray (80000,) — 5 s at 16 kHz
    → [optional] DeepFilterNet denoise (16k→48k→denoise→16k)
    → SpeakerSegmentation  → diarization SlidingWindowFeature (624, 3)
    → PixITWrapper buffers → sources tensor (1, 80000, 3)
    → WeSpeaker embedding from each separated source (normalized)
    → EnrolledSpeakerClustering    → permuted_seg SlidingWindowFeature (624, 8)
                                     + SpeakerMap (local → global)
    → source mapping: per-speaker 500 ms audio clips + activity scores
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio.functional as AF
import yaml
from pyannote.audio import Model as PyanModel
from pyannote.core import SlidingWindow, SlidingWindowFeature

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


@dataclass
class StepResult:
    """Full output of one pipeline step."""
    speakers: List[SpeakerResult] = field(default_factory=list)
    infer_ms: float = 0.0
    step_idx: int = 0


# ─── Pipeline ────────────────────────────────────────────────────

class RealtimePipeline:
    """Real-time speaker diarization + separation pipeline.

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

        # Source filtering (ghost suppression)
        sf_cfg = config.get("source_filter", {})
        self._min_peak: float = sf_cfg.get("min_peak", 0.05)
        self._ghost_ratio: float = sf_cfg.get("ghost_ratio", 0.15)

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

        # ── Embedding (WeSpeaker, direct from separated sources) ──
        log.info("Loading WeSpeaker embedding model...")
        self._emb_model = PyanModel.from_pretrained(emb_cfg["model"]).to(self.device)
        self._emb_model.eval()

        # ── DeepFilterNet denoiser (optional) — load BEFORE enrollment ──
        denoiser_cfg = config.get("denoiser", {})
        self._denoiser_enabled = denoiser_cfg.get("enabled", False)
        self._df_model = None
        self._df_state = None
        if self._denoiser_enabled:
            try:
                from df import enhance, init_df
                model_name = denoiser_cfg.get("model", "DeepFilterNet3")
                log.info("Loading DeepFilterNet (%s)...", model_name)
                self._df_model, self._df_state, _ = init_df(
                    model_base_dir=model_name
                )
                self._df_enhance = enhance
                self._df_sr = self._df_state.sr()  # 48000
                log.info("DeepFilterNet ready (native %d Hz).", self._df_sr)
            except Exception as e:
                log.warning("DeepFilterNet failed to load, disabling: %s", e)
                self._denoiser_enabled = False

        # Pre-compute resample transforms for DF (cached kernels)
        if self._denoiser_enabled:
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

        # Crossfade between consecutive steps to avoid boundary discontinuities.
        # We extract (step_samples + fade_samples) from PixIT output, then
        # crossfade the leading fade_samples with the previous step's tail.
        self._fade_samples = int(0.010 * self.sample_rate)  # 10ms = 160 samples
        self._prev_tails: Dict[int, np.ndarray] = {}  # global_idx → tail samples

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
        if not self._denoiser_enabled or self._df_model is None:
            return block
        n = len(block)
        wav_t = torch.from_numpy(block).unsqueeze(0)          # (1, samples) CPU
        wav_48 = self._df_resample_up(wav_t)                   # CPU, cached kernel
        wav_48 = self._df_enhance(self._df_model, self._df_state, wav_48)
        wav_16 = self._df_resample_down(wav_48)                # CPU, cached kernel
        return wav_16.squeeze(0).numpy()[:n]

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

            # Pad or trim to exactly chunk_samples (5s)
            if len(audio) < self.chunk_samples:
                audio = np.pad(audio, (0, self.chunk_samples - len(audio)))
            audio = audio[:self.chunk_samples]

            # Run through PixIT separation (same as live path — fp16)
            wav_gpu = torch.from_numpy(audio).to(device=self.device, dtype=torch.float16)
            wav_gpu = wav_gpu.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
            with torch.inference_mode():
                seg_t = self._pixit_wrapper(wav_gpu)
            seg_np = seg_t[0].float().cpu().numpy()             # (frames, 3)
            sources = self._pixit_wrapper.last_sources.float()  # (1, samples, 3)

            # Find the most active source (highest mean activation)
            mean_acts = np.mean(seg_np, axis=0)
            best_src = int(np.argmax(mean_acts))
            log.info(
                "  '%s' PixIT activations: %s → picking source %d",
                name, [f"{a:.3f}" for a in mean_acts], best_src,
            )

            # Extract embedding from the dominant separated source
            src_audio = sources[0, :, best_src]
            peak = src_audio.abs().max()
            if peak < 1e-4:
                log.warning("  '%s' dominant source is silent, falling back to raw", name)
                reextracted[name] = self.extract_embedding(audio)
                continue
            src_audio = src_audio / peak
            inp = src_audio.unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                emb_t = self._emb_model(inp)
            emb_t = emb_t / (emb_t.norm(dim=-1, keepdim=True) + 1e-8)
            emb = emb_t.squeeze(0).cpu().numpy().astype(np.float64)

            # Log comparison
            from scipy.spatial.distance import cosine as cos_dist
            raw_emb = self.extract_embedding(audio)
            raw_d = cos_dist(raw_emb, enrolled_embeddings[name])
            sep_d = cos_dist(emb, enrolled_embeddings[name])
            log.info(
                "  '%s' enrollment: raw→stored=%.3f, separated→stored=%.3f (DF=%s)",
                name, raw_d, sep_d,
                "ON" if self._denoiser_enabled else "OFF",
            )

            reextracted[name] = emb
            log.info("Re-extracted '%s' through %sPixIT→WeSpeaker (dim=%d)",
                     name, "DF→" if self._denoiser_enabled else "", len(emb))
        return reextracted

    def _extract_source_embedding(self, source_audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract embedding from a single separated source on GPU.

        Returns None if source is too quiet / bad quality.
        """
        # Check energy — skip silent sources
        peak = source_audio.abs().max()
        if peak < 1e-4:
            return None
        # Peak-normalize
        source_audio = source_audio / peak
        # WeSpeaker expects (batch, channels, samples)
        inp = source_audio.unsqueeze(0).unsqueeze(0)
        emb = self._emb_model(inp)  # (1, emb_dim)
        # L2-normalize
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb.squeeze(0)  # (emb_dim,)

    def step(self, waveform: np.ndarray) -> StepResult:
        """Run one pipeline step on a 5 s audio chunk.

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

        # ── 2. Batch WeSpeaker embeddings (single forward pass) ──
        active_mask = np.max(seg_np, axis=0) >= self.clustering.tau_active
        active_indices = []

        # Compute source peaks for ghost suppression
        src_peak_vals = []
        for spk_idx in range(self.n_local_speakers):
            src_peak_vals.append(float(sources[0, :, spk_idx].abs().max().cpu()))
        max_src_peak = max(src_peak_vals) if src_peak_vals else 0.0
        ghost_floor = max_src_peak * self._ghost_ratio

        # Collect and normalise active sources on GPU
        batch_tensors = []
        suppressed = []
        for spk_idx in range(self.n_local_speakers):
            if not active_mask[spk_idx]:
                continue
            peak = src_peak_vals[spk_idx]
            # Option #4: absolute energy floor
            if peak < self._min_peak:
                suppressed.append((spk_idx, f"below min_peak {peak:.4f}<{self._min_peak}"))
                active_mask[spk_idx] = False
                continue
            # Option #1: ghost ratio suppression
            if peak < ghost_floor:
                suppressed.append((spk_idx, f"ghost {peak:.4f}<{ghost_floor:.4f} ({self._ghost_ratio:.0%} of {max_src_peak:.4f})"))
                active_mask[spk_idx] = False
                continue
            src = sources[0, :, spk_idx]                 # (80000,) GPU
            batch_tensors.append((src / src.abs().max()).unsqueeze(0).unsqueeze(0))  # (1, 1, 80000)
            active_indices.append(spk_idx)

        if suppressed and self._step_idx % 10 == 0:
            log.debug("Ghost suppressed: %s", "; ".join(f"src-{s}: {r}" for s, r in suppressed))

        embeddings = torch.full((self.n_local_speakers, self._emb_dim), float('nan'))
        if batch_tensors:
            batch = torch.cat(batch_tensors, dim=0)      # (N, 1, 80000)
            with torch.inference_mode():
                embs = self._emb_model(batch)            # (N, emb_dim)
            embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-8)
            embs_cpu = embs.cpu()
            for i, spk_idx in enumerate(active_indices):
                embeddings[spk_idx] = embs_cpu[i]

        # ── DEBUG: log chunk stats + embedding distances to anchors ──
        if self._step_idx % 5 == 0 and self.clustering._anchors:
            from scipy.spatial.distance import cdist as _cdist
            chunk_peak = float(np.max(np.abs(waveform)))
            emb_np = embeddings.numpy()
            anchor_ids = sorted(self.clustering._anchors.keys())
            anchor_mat = np.array([self.clustering._anchors[a] for a in anchor_ids])
            anchor_names = [self.clustering._labels.get(a, f"g{a}") for a in anchor_ids]
            n_supp = len(suppressed)
            lines = [f"STEP-DEBUG #{self._step_idx}: chunk_peak={chunk_peak:.4f} src_peaks={[f'{p:.4f}' for p in src_peak_vals]} active={active_indices} suppressed={n_supp}"]
            for si in active_indices:
                e = emb_np[si:si+1]
                if not np.isnan(e).any():
                    dists = _cdist(e, anchor_mat, metric="cosine")[0]
                    lines.append(f"  src-{si} act={float(np.max(seg_np[:, si])):.3f} → " +
                                 ", ".join(f"{n}={d:.4f}" for n, d in zip(anchor_names, dists)))
            log.info("\n".join(lines))

        # ── 3. Clustering ──
        permuted_seg = self.clustering(segmentation, embeddings)
        speaker_map = self.clustering.last_speaker_map

        # ── 4. Map sources to global speakers ──
        n_seg_frames = permuted_seg.data.shape[0]
        step_frames = max(1, int(n_seg_frames * self.step_duration / self.duration))
        step_seg = permuted_seg.data[-step_frames:]

        # Move sources to CPU once for all speakers
        sources_np = sources[0].cpu().numpy()            # (80000, 3)

        results = []
        if speaker_map is not None:
            local_ids, global_ids = speaker_map.valid_assignments()
            for local_idx, global_idx in zip(local_ids, global_ids):
                label = self.clustering.get_label(global_idx)
                activity = float(np.mean(step_seg[:, global_idx]))

                # Extract step + fade region for overlap-add crossfade
                fade = self._fade_samples
                extract_len = self.step_samples + fade
                raw = sources_np[-extract_len:, local_idx].copy()

                # Crossfade with previous step's tail
                prev_tail = self._prev_tails.get(global_idx)
                if prev_tail is not None and len(prev_tail) == fade:
                    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                    raw[:fade] = prev_tail * (1.0 - ramp) + raw[:fade] * ramp

                # Save this step's tail for next crossfade
                self._prev_tails[global_idx] = raw[-fade:].copy()

                # Preserve the separator's natural level from step to step.
                # Per-step peak normalization made quiet chunks jump in level and
                # created audible pumping at chunk boundaries.
                audio = np.clip(raw[:self.step_samples], -1.0, 1.0)

                results.append(SpeakerResult(
                    global_idx=global_idx,
                    label=label,
                    audio=audio,
                    activity=activity,
                    is_enrolled=self.clustering.is_enrolled(global_idx),
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
        # Re-create clustering with same params, re-inject enrolled centroids
        clust_cfg = self.cfg["clustering"]
        old_labels = {
            idx: name
            for idx, name in self.clustering._labels.items()
            if self.clustering.is_enrolled(idx)
        }
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
        )
        if old_anchors:
            self.clustering.inject_centroids(old_anchors)
