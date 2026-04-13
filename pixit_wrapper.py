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
import torch.nn.functional as F
from diart.models import SegmentationModel
from huggingface_hub import HfFolder
from pyannote.audio import Model

try:
    from asteroid.utils.torch_utils import pad_x_to_y
except ImportError as exc:  # pragma: no cover - runtime dependency checked by pyannote
    raise ImportError(
        "PixIT separation requires asteroid. Install pyannote-audio[separation]."
    ) from exc


class PixITWrapper(nn.Module):
    """Wraps ToTaToNet so that forward() returns only the diarization tensor.

    The separated source waveforms and masked latent representation are stored
    after every forward pass so downstream code can reconstruct in latent space
    instead of overlap-adding already-decoded waveforms.

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
        self.last_masked_tf_rep: torch.Tensor | None = None

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Run ToTaToNet and return only the diarization activations.

        Parameters
        ----------
        waveform : torch.Tensor, shape (batch, channels, samples)

        Returns
        -------
        diarization : torch.Tensor, shape (batch, frames, speakers)
        """
        model = self.totatonet
        bsz = waveform.shape[0]
        tf_rep = model.encoder(waveform)
        if model.use_wavlm:
            wavlm_rep = model.wavlm(waveform.squeeze(1)).last_hidden_state
            wavlm_rep = wavlm_rep.transpose(1, 2)
            wavlm_rep = wavlm_rep.repeat_interleave(model.wavlm_scaling, dim=-1)
            wavlm_rep = pad_x_to_y(wavlm_rep, tf_rep)
            wavlm_rep = torch.cat((tf_rep, wavlm_rep), dim=1)
            masks = model.masker(wavlm_rep)
        else:
            masks = model.masker(tf_rep)

        masked_tf_rep = masks * tf_rep.unsqueeze(1)
        decoded_sources = self.decode_masked_tf_rep(masked_tf_rep, waveform.shape[-1])
        outputs = torch.flatten(masked_tf_rep, start_dim=0, end_dim=1)
        outputs = model.average_pool(outputs)
        outputs = outputs.transpose(1, 2)
        if model.hparams.linear["num_layers"] > 0:
            for linear in model.linear:
                outputs = F.leaky_relu(linear(outputs))
        if model.hparams.linear["num_layers"] == 0:
            outputs = (outputs**2).sum(dim=2).unsqueeze(-1)
        outputs = model.classifier(outputs)
        outputs = outputs.reshape(bsz, model.n_sources, -1)
        outputs = outputs.transpose(1, 2)
        diarization = model.activation[0](outputs)

        self.last_masked_tf_rep = masked_tf_rep.detach()
        self.last_sources = decoded_sources.detach()
        return diarization

    def decode_masked_tf_rep(
        self,
        masked_tf_rep: torch.Tensor,
        target_num_samples: int,
    ) -> torch.Tensor:
        """Decode masked latent features to waveform with causal right trim/pad."""
        decoded = self.totatonet.decoder(masked_tf_rep)
        if decoded.shape[-1] != int(target_num_samples):
            decoded = F.pad(decoded, [0, int(target_num_samples) - int(decoded.shape[-1])])
        return decoded.transpose(1, 2)


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
