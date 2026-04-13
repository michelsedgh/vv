from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


EMBEDDING_MODEL_OPTIONS: Sequence[str] = (
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "nvidia/speakerverification_en_titanet_large",
)


def embedding_backend(model_name: str) -> str:
    if model_name.startswith("nvidia/"):
        return "nemo_titanet"
    if model_name.startswith("pyannote/"):
        return "pyannote"
    raise ValueError(f"Unsupported embedding model: {model_name}")


def embedding_model_label(model_name: str) -> str:
    if model_name == "pyannote/wespeaker-voxceleb-resnet34-LM":
        return "WeSpeaker"
    if model_name == "nvidia/speakerverification_en_titanet_large":
        return "TitaNet"
    return model_name


class TitaNetEmbeddingAdapter:
    """diart-compatible wrapper around NeMo TitaNet speaker embeddings."""

    def __init__(self, model_name: str):
        from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel

        self.model_name = model_name
        self.device = torch.device("cpu")
        self.model = EncDecSpeakerLabelModel.from_pretrained(model_name)

    def to(self, device: torch.device | str) -> "TitaNetEmbeddingAdapter":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def eval(self) -> "TitaNetEmbeddingAdapter":
        self.model.eval()
        return self

    def __call__(
        self,
        waveform: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3:
            raise ValueError(
                f"Expected waveform with 2 or 3 dims, got shape {tuple(waveform.shape)}"
            )

        waveform = waveform.to(self.device, dtype=torch.float32)
        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        mono = waveform[:, 0, :]
        if weights is not None:
            if weights.ndim == 1:
                weights = weights.unsqueeze(0)
            if weights.ndim != 2:
                raise ValueError(
                    f"Expected weights with 1 or 2 dims, got shape {tuple(weights.shape)}"
                )
            weights = weights.to(self.device, dtype=torch.float32)
            weights = F.interpolate(
                weights.unsqueeze(1),
                size=mono.shape[1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)
            mono = mono * torch.clamp(weights, min=0.0)

        lengths = torch.full(
            (mono.shape[0],),
            mono.shape[1],
            dtype=torch.int64,
            device=self.device,
        )
        with torch.no_grad():
            _, embeddings = self.model.forward(
                input_signal=mono,
                input_signal_length=lengths,
            )
        return embeddings


def load_embedding_model(
    model_name: str,
    device: torch.device | str,
):
    backend = embedding_backend(model_name)
    if backend == "nemo_titanet":
        model = TitaNetEmbeddingAdapter(model_name)
    else:
        from pyannote.audio import Model as PyanModel

        model = PyanModel.from_pretrained(model_name)
    return model.to(torch.device(device)).eval()


def extract_embedding_vector(
    model,
    audio: np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-6:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)

    device = torch.device(device)
    waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    with torch.no_grad():
        if isinstance(model, TitaNetEmbeddingAdapter):
            embedding = model(waveform)
        else:
            embedding = model(waveform.unsqueeze(0))

    vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float64)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector
