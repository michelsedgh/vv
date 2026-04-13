#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.spatial.distance import cdist


BASE_DIR = Path(__file__).resolve().parent


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sample_rate != target_sr:
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = AF.resample(wav, sample_rate, target_sr)
        audio = wav.squeeze(0).numpy()
    return audio.astype(np.float32, copy=False)


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cdist(a.reshape(1, -1), b.reshape(1, -1), metric="cosine")[0, 0])


def score_bank(bank: list[np.ndarray], embedding: np.ndarray) -> float:
    return min(cosine_distance(ref, embedding) for ref in bank)


def best_threshold(positives: list[float], negatives: list[float]) -> tuple[float, float]:
    candidates = sorted({0.0, *positives, *negatives})
    best_acc = -1.0
    best_thr = 0.0
    total = len(positives) + len(negatives)
    for thr in candidates:
        correct = sum(score <= thr for score in positives)
        correct += sum(score > thr for score in negatives)
        acc = correct / total if total else 0.0
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return float(best_acc), float(best_thr)


@dataclass
class ModelRunner:
    model_name: str
    family: str
    encoder: Callable[[Path], np.ndarray]


def build_pyannote_runner(model_name: str) -> ModelRunner:
    from pyannote.audio import Model

    device = torch.device("cpu")
    model = Model.from_pretrained(model_name).to(device)
    model.eval()

    def encode(path: Path) -> np.ndarray:
        audio = load_audio(path)
        waveform = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(waveform)
        return normalize(emb.squeeze(0).cpu().numpy())

    return ModelRunner(model_name=model_name, family="pyannote", encoder=encode)


def build_speechbrain_runner(model_name: str) -> ModelRunner:
    from speechbrain.inference.speaker import EncoderClassifier

    cache_dir = BASE_DIR / "pretrained_models" / model_name.replace("/", "__")
    classifier = EncoderClassifier.from_hparams(
        source=model_name,
        savedir=str(cache_dir),
        run_opts={"device": "cpu"},
    )

    def encode(path: Path) -> np.ndarray:
        audio = load_audio(path)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        with torch.no_grad():
            emb = classifier.encode_batch(waveform)
        return normalize(emb.squeeze().cpu().numpy())

    return ModelRunner(model_name=model_name, family="speechbrain", encoder=encode)


def build_titanet_runner(model_name: str) -> ModelRunner:
    try:
        from nemo.collections.asr.models.label_models import EncDecSpeakerLabelModel
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Titanet skipped in this environment: NeMo runtime is not importable."
        ) from exc

    cache_dir = BASE_DIR / "pretrained_models" / model_name.replace("/", "__")
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = EncDecSpeakerLabelModel.from_pretrained(model_name)
    model.eval()

    def encode(path: Path) -> np.ndarray:
        with torch.no_grad():
            emb = model.get_embedding(str(path))
        if hasattr(emb, "cpu"):
            emb = emb.cpu().numpy()
        return normalize(np.asarray(emb))

    return ModelRunner(model_name=model_name, family="nemo", encoder=encode)


def discover_dataset(speaker: str) -> dict[str, list[Path] | Path]:
    speaker_dir = BASE_DIR / "enrolled_speakers" / speaker
    close_ref = speaker_dir / "reference.wav"
    far_ref = speaker_dir / "reference_far.wav"
    if not close_ref.exists():
        raise FileNotFoundError(f"Missing close reference: {close_ref}")
    if not far_ref.exists():
        raise FileNotFoundError(f"Missing far reference: {far_ref}")

    positive_patterns = [
        BASE_DIR / "diagnostic_clips" / "michel_*.wav",
    ]
    positives: list[Path] = []
    for pattern in positive_patterns:
        positives.extend(sorted(pattern.parent.glob(pattern.name)))

    negatives = sorted(
        path
        for path in (BASE_DIR / "diagnostic_clips").glob("*.wav")
        if not path.name.startswith("michel_")
    )
    if not positives:
        raise RuntimeError("No held-out positive clips found")
    if not negatives:
        raise RuntimeError("No negative clips found")
    return {
        "close_ref": close_ref,
        "far_ref": far_ref,
        "positives": positives,
        "negatives": negatives,
    }


def benchmark_model(runner: ModelRunner, dataset: dict[str, list[Path] | Path]) -> dict:
    all_paths: list[Path] = [
        dataset["close_ref"],  # type: ignore[list-item]
        dataset["far_ref"],  # type: ignore[list-item]
        *dataset["positives"],  # type: ignore[list-item]
        *dataset["negatives"],  # type: ignore[list-item]
    ]
    seen: dict[Path, np.ndarray] = {}
    encode_ms: list[float] = []
    for path in all_paths:
        if path in seen:
            continue
        start = time.perf_counter()
        seen[path] = runner.encoder(path)
        encode_ms.append((time.perf_counter() - start) * 1000.0)

    close_vec = seen[dataset["close_ref"]]  # type: ignore[index]
    far_vec = seen[dataset["far_ref"]]  # type: ignore[index]
    positives = [seen[path] for path in dataset["positives"]]  # type: ignore[index]
    negatives = [seen[path] for path in dataset["negatives"]]  # type: ignore[index]

    close_only_bank = [close_vec]
    multi_bank = [close_vec, far_vec]

    close_positive_scores = [score_bank(close_only_bank, emb) for emb in [far_vec, *positives]]
    close_negative_scores = [score_bank(close_only_bank, emb) for emb in negatives]
    multi_positive_scores = [score_bank(multi_bank, emb) for emb in positives]
    multi_negative_scores = [score_bank(multi_bank, emb) for emb in negatives]

    close_acc, close_thr = best_threshold(close_positive_scores, close_negative_scores)
    multi_acc, multi_thr = best_threshold(multi_positive_scores, multi_negative_scores)
    return {
        "model": runner.model_name,
        "family": runner.family,
        "status": "ok",
        "error": None,
        "close_far_distance": cosine_distance(close_vec, far_vec),
        "close_only_best_acc": close_acc,
        "close_only_threshold": close_thr,
        "close_only_max_positive": float(max(close_positive_scores)),
        "close_only_min_negative": float(min(close_negative_scores)),
        "close_only_margin": float(min(close_negative_scores) - max(close_positive_scores)),
        "multi_best_acc": multi_acc,
        "multi_threshold": multi_thr,
        "multi_max_positive": float(max(multi_positive_scores)),
        "multi_min_negative": float(min(multi_negative_scores)),
        "multi_margin": float(min(multi_negative_scores) - max(multi_positive_scores)),
        "avg_encode_ms": float(sum(encode_ms) / len(encode_ms)),
        "embedding_dim": int(close_vec.shape[0]),
        "num_positive_eval": len(positives),
        "num_negative_eval": len(negatives),
        "positive_files": [path.name for path in dataset["positives"]],  # type: ignore[index]
        "negative_files": [path.name for path in dataset["negatives"]],  # type: ignore[index]
    }


def make_runner(model_name: str) -> ModelRunner:
    if model_name.startswith("speechbrain/"):
        return build_speechbrain_runner(model_name)
    if model_name.startswith("nvidia/"):
        return build_titanet_runner(model_name)
    return build_pyannote_runner(model_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark speaker embedding models on local room clips")
    parser.add_argument("--speaker", default="mich")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model to benchmark. Repeat to benchmark multiple models.",
    )
    args = parser.parse_args()

    models = args.models or [
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "speechbrain/spkrec-ecapa-voxceleb",
        "nvidia/speakerverification_en_titanet_large",
    ]
    dataset = discover_dataset(args.speaker)

    results = []
    for model_name in models:
        try:
            runner = make_runner(model_name)
            results.append(benchmark_model(runner, dataset))
        except Exception as exc:  # pragma: no cover - environment dependent
            family = (
                "speechbrain" if model_name.startswith("speechbrain/")
                else "nemo" if model_name.startswith("nvidia/")
                else "pyannote"
            )
            results.append({
                "model": model_name,
                "family": family,
                "status": "error",
                "error": str(exc),
                "close_far_distance": None,
                "close_only_best_acc": None,
                "close_only_threshold": None,
                "close_only_max_positive": None,
                "close_only_min_negative": None,
                "close_only_margin": None,
                "multi_best_acc": None,
                "multi_threshold": None,
                "multi_max_positive": None,
                "multi_min_negative": None,
                "multi_margin": None,
                "avg_encode_ms": None,
                "embedding_dim": None,
            })

    out_path = BASE_DIR / "logs" / f"embedding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"saved_to": str(out_path), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
