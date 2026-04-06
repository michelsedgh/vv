"""
PixIT (ToTaToNet) wrapper for diart compatibility.

ToTaToNet.forward() returns a tuple (diarization, sources), but diart's
SegmentationModel expects a single tensor (batch, frames, speakers).

This module provides:
  - PixITWrapper: nn.Module that returns only the diarization tensor and
    side-buffers the separated sources for later retrieval.
  - make_pixit_segmentation_model(): factory that creates a diart
    SegmentationModel backed by PixIT via a custom lazy loader.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from diart.models import SegmentationModel
from huggingface_hub import HfFolder
from pyannote.audio import Model


class PixITWrapper(nn.Module):
    """Wraps ToTaToNet so that forward() returns only the diarization tensor.

    The separated source waveforms are stored in ``last_sources`` after every
    forward pass so that downstream code (pipeline.py) can retrieve them.

    Shapes (for a 5 s chunk at 16 kHz):
        input  : (batch, 1, 80000)
        diarization : (batch, 624, 3)   — sigmoid speaker activity
        sources     : (batch, 80000, 3) — separated waveforms
    """

    def __init__(self, totatonet: nn.Module):
        super().__init__()
        self.totatonet = totatonet
        # Populated after every forward(); detached, on the same device.
        self.last_sources: torch.Tensor | None = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run ToTaToNet and return only the diarization activations.

        Parameters
        ----------
        waveform : torch.Tensor, shape (batch, channels, samples)

        Returns
        -------
        diarization : torch.Tensor, shape (batch, frames, speakers)
        """
        diarization, sources = self.totatonet(waveform)
        # Detach sources so they don't hold the compute graph in memory.
        self.last_sources = sources.detach()
        return diarization


def make_pixit_segmentation_model(
    model_id: str = "pyannote/separation-ami-1.0",
) -> SegmentationModel:
    """Create a diart ``SegmentationModel`` backed by PixIT.

    Uses a custom lazy loader that bypasses ``PyannoteLoader`` (which would
    crash on PixIT's tuple specifications).

    Parameters
    ----------
    model_id : str
        Hugging Face model identifier for the ToTaToNet checkpoint.

    Returns
    -------
    SegmentationModel
        A lazily-loaded segmentation model wrapping PixIT.
    """

    def _loader() -> PixITWrapper:
        token = HfFolder.get_token()
        raw_model = Model.from_pretrained(model_id, use_auth_token=token)
        if raw_model is None:
            raise RuntimeError(
                f"Failed to load {model_id}. "
                "Accept the model conditions at https://hf.co/pyannote/separation-ami-1.0"
            )
        return PixITWrapper(raw_model)

    return SegmentationModel(_loader)
