"""
TensorRT-accelerated WavLM encoder replacement.

Loads a pre-built TRT engine for the WavLM transformer encoder and provides
a drop-in replacement that runs: PyTorch CNN → PyTorch projection → TRT encoder.

Usage:
    If engines/wavlm_encoder_fp16.engine exists, pipeline.py will auto-detect
    and use it via `patch_wavlm_with_trt(wavlm_model)`.
"""
from __future__ import annotations

import logging
import os
import gc

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

log = logging.getLogger(__name__)

ENGINE_PATH = os.path.join(os.path.dirname(__file__), "engines", "wavlm_encoder_fp16.engine")


def trt_engine_available() -> bool:
    """Check if the pre-built TRT engine exists."""
    return os.path.isfile(ENGINE_PATH)


class TRTEncoderRunner:
    """Runs the WavLM transformer encoder via TensorRT.

    Allocates persistent GPU I/O buffers and executes the TRT engine
    using PyTorch's CUDA stream for synchronization.
    """

    def __init__(self, engine_path: str, device: torch.device):
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        log.info("Loading TRT engine from %s (%.0f MB)...",
                 engine_path, os.path.getsize(engine_path) / 1024**2)

        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

        # Discover I/O tensor names, shapes, and dtypes
        self._io_info = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            mode = self.engine.get_tensor_mode(name)
            self._io_info[name] = {
                "shape": shape,
                "trt_dtype": trt_dtype,
                "mode": mode,
                "torch_dtype": self._trt_to_torch_dtype(trt_dtype),
            }
            log.info("  TRT tensor: %s shape=%s dtype=%s mode=%s",
                     name, shape, trt_dtype, mode)

        # Allocate persistent I/O buffers on GPU
        self._buffers = {}
        for name, info in self._io_info.items():
            buf = torch.empty(info["shape"], dtype=info["torch_dtype"], device=device)
            self._buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())

        log.info("TRT encoder ready.")

    @staticmethod
    def _trt_to_torch_dtype(trt_dtype):
        import tensorrt as trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int8: torch.int8,
            trt.int32: torch.int32,
        }
        return mapping.get(trt_dtype, torch.float32)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run TRT encoder. Input/output: (1, seq_len, hidden_dim)."""
        inp_name = "features"
        out_name = "hidden_states"
        inp_info = self._io_info[inp_name]
        out_info = self._io_info[out_name]

        # Copy input to the engine's input buffer (with dtype cast if needed)
        inp_buf = self._buffers[inp_name]
        if hidden_states.dtype != inp_info["torch_dtype"]:
            inp_buf.copy_(hidden_states.to(inp_info["torch_dtype"]))
        else:
            inp_buf.copy_(hidden_states)

        # Execute on our stream
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)

        # Return output (cast back to input dtype if needed)
        out_buf = self._buffers[out_name]
        if out_buf.dtype != hidden_states.dtype:
            return out_buf.to(hidden_states.dtype)
        return out_buf.clone()


class TRTWavLMEncoder(nn.Module):
    """Drop-in replacement for WavLM's encoder that uses TRT.

    Wraps TRTEncoderRunner and returns a FakeEncoderOutput with
    .last_hidden_state, just like the HuggingFace WavLMEncoder.
    """

    def __init__(self, trt_runner: TRTEncoderRunner):
        super().__init__()
        self.trt_runner = trt_runner

    def forward(self, hidden_states, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        out = self.trt_runner(hidden_states)
        return BaseModelOutput(last_hidden_state=out, hidden_states=None, attentions=None)


def patch_wavlm_with_trt(wavlm_model: nn.Module, device: torch.device) -> bool:
    """Replace WavLM's encoder with the TRT version.

    Parameters
    ----------
    wavlm_model : nn.Module
        The WavLMModel instance (e.g., pixit.totatonet.wavlm)
    device : torch.device
        CUDA device

    Returns
    -------
    bool
        True if patching succeeded, False if engine not available or failed.
    """
    if not trt_engine_available():
        log.info("TRT engine not found at %s — using PyTorch encoder.", ENGINE_PATH)
        return False

    try:
        # Free PyTorch encoder FIRST to make room for TRT engine (~600 MB each)
        old_encoder = wavlm_model.encoder
        old_params_mb = sum(p.numel() * p.element_size() for p in old_encoder.parameters()) / 1024**2
        log.info("Freeing PyTorch encoder (%.0f MB) to make room for TRT...", old_params_mb)

        # Replace with a dummy so wavlm_model no longer holds references
        wavlm_model.encoder = nn.Identity()
        del old_encoder
        gc.collect()
        torch.cuda.empty_cache()

        # Now load TRT engine into the freed memory
        runner = TRTEncoderRunner(ENGINE_PATH, device)
        wavlm_model.encoder = TRTWavLMEncoder(runner)
        log.info("WavLM encoder replaced with TRT (%.0f MB freed → TRT loaded).", old_params_mb)
        return True

    except Exception as e:
        log.warning("TRT encoder patch failed: %s — falling back to PyTorch.", e)
        return False
