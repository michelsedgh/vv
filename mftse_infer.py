"""
MeanFlow-TSE inference wrapper for live microphone chunks.
Adds meanflow_tse/ to sys.path and reuses eval_steps sampling + STFT utils.
Matches eval_steps.py ECAPAMLP path: alpha = t_predicter(mix_wav, enroll_wav) then STFT → Euler.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

VOICE_ROOT = Path(__file__).resolve().parent
MF_ROOT = VOICE_ROOT / "meanflow_tse"
if str(MF_ROOT) not in sys.path:
    sys.path.insert(0, str(MF_ROOT))

from eval_steps import (  # noqa: E402
    pad_and_reshape,
    reshape_and_remove_padding,
    sample_euler_multistep,
    scale_audio,
)
from train_meanflow import LightningModule as MeanFlowLightningModule  # noqa: E402
from train_meanflow import parse_config as parse_meanflow_config  # noqa: E402
from utils.transforms import istft_torch, stft_torch  # noqa: E402


def load_t_predictor_from_lightning_ckpt(
    ckpt_path: Path,
    t_pred_cfg: dict,
    device: torch.device,
) -> Any:
    """
    Load TPredicter weights from train_t_predicter.py Lightning checkpoint.
    Same key stripping as meanflow_tse/inference_sample.ipynb.
    """
    from models.t_predicter import TPredicter

    ckpt_path = Path(ckpt_path)
    load_kw = {"map_location": device}
    try:
        checkpoint = torch.load(str(ckpt_path), **load_kw, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), **load_kw)
    state_dict = checkpoint["state_dict"]
    new_sd = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            new_sd[key[6:]] = value
        else:
            new_sd[key] = value
    net = TPredicter(**t_pred_cfg)
    net.load_state_dict(new_sd)
    net.eval()
    return net.to(device)


def to_mono_float32(wave: np.ndarray) -> np.ndarray:
    x = np.asarray(wave, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and x.shape[1] >= 2:
        return x.mean(axis=1).astype(np.float32)
    return x.reshape(-1).astype(np.float32)


def fix_length(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n].copy()
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out


class MeanFlowTSERunner:
    def __init__(
        self,
        base_config_path: Path,
        checkpoint_path: Path,
        *,
        device: Optional[str] = None,
        use_t_predictor: bool = True,
        t_predictor_checkpoint: Optional[Path] = None,
    ):
        self.base_config_path = Path(base_config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.config = parse_meanflow_config(str(self.base_config_path))
        self.config["eval"]["checkpoint"] = str(self.checkpoint_path.resolve())

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"MeanFlow checkpoint not found: {self.checkpoint_path}\n"
                "Download from the MeanFlow-TSE README (Google Drive) into models/meanflow/"
            )

        print(f"[MeanFlow-TSE] Loading {self.checkpoint_path} on {self.device}...", flush=True)
        self.pl_module = MeanFlowLightningModule.load_from_checkpoint(
            str(self.checkpoint_path),
            config=self.config,
        )
        self.pl_module.eval()
        self.pl_module.to(self.device)

        # Disable gradient checkpointing — only useful during training,
        # adds overhead at inference and breaks torch.compile.
        for module in self.pl_module.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = False

        self.t_predictor: Optional[Any] = None
        if use_t_predictor:
            t_ckpt = Path(t_predictor_checkpoint) if t_predictor_checkpoint else None
            if t_ckpt is None or not t_ckpt.is_file():
                raise FileNotFoundError(
                    "meanflow.use_t_predictor is true but t_predictor_checkpoint is missing or not a file.\n"
                    f"Expected: {t_ckpt}"
                )
            print(f"[MeanFlow-TSE] Loading t-predictor {t_ckpt}...", flush=True)
            self.t_predictor = load_t_predictor_from_lightning_ckpt(
                t_ckpt,
                self.config["t_predicter"],
                self.device,
            )
        else:
            print("[MeanFlow-TSE] t-predictor disabled; using alpha=0.5 (HALF).", flush=True)

        self.ds = self.config["dataset"]
        self.n_fft = int(self.ds["n_fft"])
        self.hop_length = int(self.ds["hop_length"])
        self.win_length = int(self.ds["win_length"])
        self.sample_rate = int(self.ds["sample_rate"])
        # pad_and_reshape tile width (STFT frames) must match eval_steps / training — NOT live chunk_seconds.
        # eval always uses segment-length tiles (e.g. 3s → 376 frames); shorter mic chunks get spectrogram
        # zero-pad to this width before the UDiT. Using chunk-sized _multiple broke enroll|mix time layout
        # (train ~752 STFT positions vs ~502 for 1s chunks) and caused silence/garbage between segments.
        seg_sec = float(self.ds.get("segment", 3))
        seg_samples = int(round(seg_sec * self.sample_rate))
        self._multiple = seg_samples // self.hop_length + 1

        # ── Optimization state ─────────────────────────────────────
        self._use_amp = False
        self._amp_dtype = torch.float16
        self._compiled = False

        # ── Enrollment cache (STFT + wav tensor, reused every chunk) ──
        self._cached_enroll_spec: Optional[torch.Tensor] = None
        self._cached_enroll_wav_t: Optional[torch.Tensor] = None

    # ── Optimization methods ──────────────────────────────────────

    def enable_amp(self, dtype: torch.dtype = torch.float16):
        """Enable torch.autocast for mixed-precision inference (FP16/BF16).
        STFT/ISTFT stay float32; only the UDiT + t-predictor run in lower precision."""
        if self.device.type != "cuda":
            print("[MeanFlow-TSE] AMP skipped (CPU device).", flush=True)
            return
        self._use_amp = True
        self._amp_dtype = dtype
        print(f"[MeanFlow-TSE] AMP enabled ({dtype}).", flush=True)

    def compile_model(self):
        """Apply torch.compile for faster inference.
        Uses 'default' mode (no triton dependency). Falls back gracefully on error."""
        if self.device.type != "cuda":
            print("[MeanFlow-TSE] torch.compile skipped (CPU device).", flush=True)
            return
        try:
            # Suppress dynamo errors → falls back to eager if compilation fails at runtime
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            # 'default' mode: no triton/CUDA-graph dependency (reduce-overhead needs triton).
            self.pl_module.model = torch.compile(
                self.pl_module.model, mode="default"
            )
            self._compiled = True
            print("[MeanFlow-TSE] torch.compile applied (default mode, suppress_errors=True).", flush=True)
        except Exception as e:
            print(f"[MeanFlow-TSE] torch.compile failed, continuing without: {e}", flush=True)

    def warmup(self, enroll_samples: int, num_steps: int = 1):
        """Run one dummy inference to trigger torch.compile JIT and CUDA warm-up.
        Call AFTER cache_enrollment() and enable_amp()/compile_model()."""
        seg_samples = int(round(float(self.ds.get("segment", 3)) * self.sample_rate))
        dummy_mix = np.zeros(seg_samples, dtype=np.float32)
        dummy_enr = np.zeros(enroll_samples, dtype=np.float32)
        print("[MeanFlow-TSE] Warmup inference (may be slow if compiled)...", flush=True)
        t0 = time.perf_counter()
        self.infer_chunk(dummy_mix, dummy_enr, num_steps=num_steps)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[MeanFlow-TSE] Warmup done in {dt:.0f} ms.", flush=True)

    # ── Enrollment caching ────────────────────────────────────────

    def cache_enrollment(self, enroll_wav: np.ndarray):
        """Pre-compute enrollment STFT and wav tensor.  Called once at monitor start;
        reused for every subsequent infer_chunk call (saves ~5-10% per chunk)."""
        enr = to_mono_float32(enroll_wav)
        enr_t = torch.from_numpy(enr).float().unsqueeze(0).to(self.device)
        enroll_spec = stft_torch(
            enr_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )
        self._cached_enroll_wav_t = enr_t
        self._cached_enroll_spec = enroll_spec
        print(
            f"[MeanFlow-TSE] Enrollment cached: wav {enr_t.shape}, spec {enroll_spec.shape}.",
            flush=True,
        )

    def clear_enrollment_cache(self):
        self._cached_enroll_spec = None
        self._cached_enroll_wav_t = None

    # ── Core inference ────────────────────────────────────────────

    @torch.inference_mode()
    def infer_chunk(
        self,
        mixture_wav: np.ndarray,
        enroll_wav: np.ndarray,
        *,
        num_steps: int = 1,
    ) -> np.ndarray:
        """
        mixture_wav, enroll_wav: 1D float32 ~[-1,1], same sample_rate as dataset (16 kHz).
        enroll_wav length should match training segment_aux (enroll_seconds * sr).
        Returns 1D float32 waveform same length as mixture (after ISTFT trim).
        """
        mix = to_mono_float32(mixture_wav)
        T = len(mix)
        mix_t = torch.from_numpy(mix).float().unsqueeze(0).to(self.device)

        # ── Use cached enrollment or compute fresh ──
        if self._cached_enroll_spec is not None:
            enroll_spec = self._cached_enroll_spec
            enr_t = self._cached_enroll_wav_t
        else:
            enr = to_mono_float32(enroll_wav)
            enr_t = torch.from_numpy(enr).float().unsqueeze(0).to(self.device)
            enroll_spec = stft_torch(
                enr_t,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
            )

        # ── t-predictor: alpha from raw waveforms ──
        if self.t_predictor is not None:
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._use_amp,
            ):
                alpha = self.t_predictor(mix_t, enr_t, aug=False)
            alpha = alpha.float()
            if alpha.ndim == 0:
                alpha = alpha.unsqueeze(0)
            alpha = alpha.reshape(-1).to(self.device)
            alpha = alpha.clamp(1e-3, 1.0 - 1e-3)
        else:
            alpha = torch.tensor([0.5], device=self.device, dtype=torch.float32)

        # ── STFT (always float32 for numerical stability) ──
        mixture_spec = stft_torch(
            mix_t,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        mixture_spec, original_length = pad_and_reshape(mixture_spec, self._multiple)
        batch_size = mixture_spec.shape[0]
        enroll_rep = enroll_spec.repeat(batch_size, 1, 1)

        # ── UDiT forward (optionally in FP16 via autocast) ──
        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._use_amp,
        ):
            source_hat_spec = sample_euler_multistep(
                model=self.pl_module.model,
                mixture_spec=mixture_spec.float(),
                enrollment_spec=enroll_rep,
                alpha=alpha,
                num_steps=num_steps,
            )

        # ── Back to float32 for ISTFT ──
        source_hat_spec = source_hat_spec.float()
        source_hat_spec = reshape_and_remove_padding(source_hat_spec, original_length)
        source_hat = istft_torch(
            source_hat_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            length=T,
        )
        source_hat = scale_audio(source_hat.cpu())
        out = source_hat.squeeze(0).numpy().astype(np.float32)
        if not np.isfinite(out).all():
            return mix.copy()
        return out
