#!/usr/bin/env python3
"""
Speaker Enrollment Script
Records audio from a microphone and saves speaker embeddings for verification.
Usage:
    python enroll_speaker.py --name "Michel" --duration 10
    python enroll_speaker.py --name "Michel" --wav /path/to/audio.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import yaml
from speechbrain.inference.speaker import EncoderClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def record_audio(duration, sample_rate, device_index=None):
    """Record audio from microphone."""
    print(f"\n  Recording for {duration} seconds...")
    print("  Speak clearly and naturally.\n")
    time.sleep(0.5)

    print("  >>> RECORDING NOW <<<")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    print("  >>> DONE <<<\n")

    return audio.squeeze()


def extract_embedding(classifier, audio, sample_rate):
    """Extract speaker embedding from audio using ECAPA-TDNN."""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    with torch.no_grad():
        embedding = classifier.encode_batch(audio)

    # Shape: [1, 1, 192] -> [192]
    return embedding.squeeze().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker for voice verification")
    parser.add_argument("--name", required=True, help="Speaker name (e.g., 'Michel')")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (default: 10)")
    parser.add_argument("--wav", type=str, default=None, help="Path to existing WAV file instead of recording")
    parser.add_argument("--samples", type=int, default=3, help="Number of enrollment samples (default: 3)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device-index", type=int, default=None, help="Audio device index (override config)")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_rate = config["audio"]["sample_rate"]
    device_index = args.device_index or config["audio"]["device_index"]
    enrollment_dir = Path(config["verification"]["enrollment_dir"])
    compute_device = config["device"]

    # Create enrollment directory
    speaker_dir = enrollment_dir / args.name.lower().replace(" ", "_")
    speaker_dir.mkdir(parents=True, exist_ok=True)

    # Load ECAPA-TDNN model
    print(f"Loading embedding model on {compute_device}...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb"),
        run_opts={"device": compute_device},
    )
    print("Model loaded.\n")

    embeddings = []

    if args.wav:
        # Load from existing WAV file
        print(f"Loading audio from: {args.wav}")
        audio, sr = sf.read(args.wav)
        if sr != sample_rate:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, sample_rate)
            audio = audio_tensor.squeeze().numpy()

        emb = extract_embedding(classifier, audio, sample_rate)
        embeddings.append(emb)
        print(f"  Extracted embedding from WAV file (dim={emb.shape[0]})")
    else:
        # Record multiple samples
        print("=" * 50)
        print(f"Enrolling speaker: {args.name}")
        print(f"Will record {args.samples} samples of {args.duration}s each.")
        print("=" * 50)

        # List available devices
        print("\nAvailable audio devices:")
        print(sd.query_devices())
        if device_index is not None:
            print(f"\nUsing device index: {device_index}")
        print()

        for i in range(args.samples):
            print(f"--- Sample {i+1}/{args.samples} ---")
            input("  Press ENTER when ready to record...")

            audio = record_audio(args.duration, sample_rate, device_index)

            # Save the wav file for reference
            wav_path = speaker_dir / f"sample_{i+1}.wav"
            sf.write(str(wav_path), audio, sample_rate)
            print(f"  Saved audio to: {wav_path}")

            # Extract embedding
            emb = extract_embedding(classifier, audio, sample_rate)
            embeddings.append(emb)
            print(f"  Extracted embedding (dim={emb.shape[0]})")

    # Average all embeddings for a robust speaker profile
    avg_embedding = np.mean(embeddings, axis=0)
    # L2 normalize
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    # Save the embedding
    emb_path = speaker_dir / "embedding.npy"
    np.save(str(emb_path), avg_embedding)
    print(f"\n{'=' * 50}")
    print(f"Enrollment complete for: {args.name}")
    print(f"  Embedding saved to: {emb_path}")
    print(f"  Embedding dimension: {avg_embedding.shape[0]}")
    print(f"  Number of samples averaged: {len(embeddings)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
