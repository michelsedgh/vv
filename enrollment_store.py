from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.spatial.distance import cdist

from embedding_runtime import extract_embedding_vector, load_embedding_model


def reference_paths(speaker_dir: Path) -> List[Path]:
    return sorted(
        path for path in speaker_dir.glob("reference*.wav") if path.is_file()
    )


def iter_reference_speaker_dirs(enroll_dir: Path) -> List[Path]:
    if not enroll_dir.exists():
        return []
    return [
        speaker_dir
        for speaker_dir in sorted(enroll_dir.iterdir())
        if speaker_dir.is_dir() and reference_paths(speaker_dir)
    ]


def embedding_cache_path(speaker_dir: Path) -> Path:
    return speaker_dir / "embedding.npy"


def _load_reference_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sample_rate != target_sr:
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = AF.resample(wav, sample_rate, target_sr)
        audio = wav.squeeze(0).numpy()
    return audio.astype(np.float32, copy=False)


def _medoid_embedding(rows: np.ndarray) -> np.ndarray:
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("Expected at least one embedding row")
    if rows.shape[0] == 1:
        anchor = rows[0].copy()
    else:
        dist_matrix = cdist(rows, rows, metric="cosine")
        medoid_idx = min(
            range(rows.shape[0]),
            key=lambda idx: float(dist_matrix[idx].mean()),
        )
        anchor = rows[medoid_idx].copy()
    norm = np.linalg.norm(anchor)
    if norm > 0:
        anchor = anchor / norm
    return anchor.astype(np.float64, copy=False)


def rebuild_speaker_embedding_cache(
    speaker_dir: Path,
    cfg: dict,
    *,
    model=None,
    device: Optional[torch.device] = None,
) -> Optional[np.ndarray]:
    refs = reference_paths(speaker_dir)
    cache_path = embedding_cache_path(speaker_dir)
    if not refs:
        if cache_path.exists():
            cache_path.unlink()
        return None

    own_model = model is None
    device = torch.device(device or cfg["device"])
    if own_model:
        model = load_embedding_model(cfg["embedding"]["model"], device)

    try:
        ref_embeddings: List[np.ndarray] = []
        sample_rate = int(cfg["audio"]["sample_rate"])
        for ref_path in refs:
            audio = _load_reference_audio(ref_path, sample_rate)
            ref_embeddings.append(
                extract_embedding_vector(model, audio, device)
            )
        if not ref_embeddings:
            if cache_path.exists():
                cache_path.unlink()
            return None
        anchor = _medoid_embedding(np.stack(ref_embeddings, axis=0))
        np.save(cache_path, anchor)
        return anchor
    finally:
        if own_model:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def rebuild_enrollment_cache(enroll_dir: Path, cfg: dict) -> Dict[str, np.ndarray]:
    enrolled: Dict[str, np.ndarray] = {}
    if not enroll_dir.exists():
        return enrolled

    device = torch.device(cfg["device"])
    model = load_embedding_model(cfg["embedding"]["model"], device)
    try:
        for speaker_dir in sorted(enroll_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            embedding = rebuild_speaker_embedding_cache(
                speaker_dir,
                cfg,
                model=model,
                device=device,
            )
            if embedding is not None:
                enrolled[speaker_dir.name] = embedding
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return enrolled
