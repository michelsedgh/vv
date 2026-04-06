"""
ONNX Runtime-accelerated DPRNN blocks replacement.

Replaces the 6 DPRNNBlock Sequential in asteroid's DPRNN masker with an
ONNX Runtime session running on CUDA EP.

Usage:
    If engines/dprnn_blocks_fp16.onnx exists, pipeline.py will auto-detect
    and use it via `patch_masker_with_ort(masker)`.
"""
from __future__ import annotations

import gc
import logging
import os

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

ONNX_PATH = os.path.join(os.path.dirname(__file__), "engines", "dprnn_blocks_fp16_safe.onnx")


def ort_dprnn_available() -> bool:
    return os.path.isfile(ONNX_PATH)


class ORTDPRNNBlocks(nn.Module):
    """Drop-in replacement for masker.net (Sequential of 6 DPRNNBlocks).

    Uses ONNX Runtime CUDA EP with mixed fp16/fp32 ONNX (norm ops in fp32,
    LSTM/linear in fp16). fp32 IO binding, ~2x speedup over PyTorch.
    Input/output shape: (1, 128, 100, n_chunks) on GPU.
    """

    def __init__(self, onnx_path: str, device: torch.device):
        super().__init__()
        import onnxruntime as ort

        log.info("Loading ORT DPRNN from %s (%.1f MB)...",
                 onnx_path, os.path.getsize(onnx_path) / 1024**2)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=[("CUDAExecutionProvider", {"device_id": 0})],
        )
        self.device = device

        # Pre-allocate persistent GPU buffers (fp32 for ORT, model runs fp16)
        # Shape: (1, 128, 100, 102) — fixed for 5s@16kHz with chunk_size=100, hop=50
        self._inp_buf = torch.empty(1, 128, 100, 102, dtype=torch.float32, device=device)
        self._out_buf = torch.empty(1, 128, 100, 102, dtype=torch.float32, device=device)

        # Pre-build IO binding
        self._binding = self.session.io_binding()
        self._binding.bind_input(
            "x", "cuda", 0, np.float32,
            list(self._inp_buf.shape), self._inp_buf.data_ptr(),
        )
        self._binding.bind_output(
            "out", "cuda", 0, np.float32,
            list(self._out_buf.shape), self._out_buf.data_ptr(),
        )

        log.info("ORT DPRNN ready (CUDA EP, fp32 IO binding).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast fp16 → fp32 for ORT, run, cast back
        self._inp_buf.copy_(x.float())
        self.session.run_with_iobinding(self._binding)
        return self._out_buf.to(x.dtype).clone()


def patch_masker_with_ort(masker: nn.Module, device: torch.device) -> bool:
    """Replace masker.net (6 DPRNNBlocks) with ORT-accelerated version.

    Parameters
    ----------
    masker : nn.Module
        The asteroid DPRNN masker (must have .net attribute).
    device : torch.device
        CUDA device.

    Returns
    -------
    bool
        True if patching succeeded.
    """
    if not ort_dprnn_available():
        log.info("ORT DPRNN ONNX not found at %s — using PyTorch.", ONNX_PATH)
        return False

    try:
        # Free PyTorch DPRNN blocks first
        old_net = masker.net
        old_params_mb = sum(p.numel() * p.element_size() for p in old_net.parameters()) / 1024**2
        log.info("Freeing PyTorch DPRNN blocks (%.0f MB)...", old_params_mb)

        masker.net = nn.Identity()
        del old_net
        gc.collect()
        torch.cuda.empty_cache()

        # Load ORT replacement
        ort_blocks = ORTDPRNNBlocks(ONNX_PATH, device)
        masker.net = ort_blocks
        log.info("DPRNN blocks replaced with ORT (%.0f MB freed).", old_params_mb)
        return True

    except Exception as e:
        log.warning("ORT DPRNN patch failed: %s — falling back to PyTorch.", e)
        return False
