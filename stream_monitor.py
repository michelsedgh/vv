#!/usr/bin/env python3
"""
24/7 Real-Time Speaker Diarization + Verification
Continuously monitors microphone, diarizes speakers, and verifies them
against enrolled speaker profiles.

Usage:
    python stream_monitor.py
    python stream_monitor.py --config config.yaml
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.inference import StreamingInference
from diart.models import EmbeddingModel, SegmentationModel
from diart.sources import MicrophoneAudioSource
from pyannote.core import Annotation, SlidingWindowFeature
from speechbrain.inference.speaker import EncoderClassifier


# ─── Configuration ───────────────────────────────────────────────────
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── Enrolled Speaker Database ───────────────────────────────────────
class SpeakerDatabase:
    """Loads and manages enrolled speaker embeddings."""

    def __init__(self, enrollment_dir, threshold=0.45):
        self.enrollment_dir = Path(enrollment_dir)
        self.threshold = threshold
        self.speakers = {}  # name -> normalized embedding
        self.load_enrolled_speakers()

    def load_enrolled_speakers(self):
        """Load all enrolled speaker embeddings from disk."""
        if not self.enrollment_dir.exists():
            print(f"[WARN] Enrollment directory not found: {self.enrollment_dir}")
            print("[WARN] No enrolled speakers. Run enroll_speaker.py first.")
            return

        for speaker_dir in sorted(self.enrollment_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue
            emb_path = speaker_dir / "embedding.npy"
            if emb_path.exists():
                emb = np.load(str(emb_path))
                emb = emb / np.linalg.norm(emb)  # ensure L2-normalized
                name = speaker_dir.name.replace("_", " ").title()
                self.speakers[name] = emb
                print(f"  Loaded enrolled speaker: {name}")

        print(f"  Total enrolled speakers: {len(self.speakers)}")

    def identify(self, embedding):
        """
        Compare an embedding against all enrolled speakers.
        Returns (name, similarity) of best match, or (None, 0.0) if below threshold.
        """
        if len(self.speakers) == 0:
            return None, 0.0

        embedding = embedding / np.linalg.norm(embedding)
        best_name = None
        best_sim = -1.0

        for name, enrolled_emb in self.speakers.items():
            sim = float(np.dot(embedding, enrolled_emb))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= self.threshold:
            return best_name, best_sim
        else:
            return None, best_sim


# ─── Verification Layer ─────────────────────────────────────────────
class SpeakerVerifier:
    """Extracts embeddings from audio segments and verifies speakers."""

    def __init__(self, database, compute_device="cuda"):
        self.database = database
        self.device = compute_device
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.expanduser("~/.cache/speechbrain/spkrec-ecapa-voxceleb"),
            run_opts={"device": compute_device},
        )

    def extract_embedding(self, audio_chunk):
        """Extract embedding from a numpy audio chunk."""
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk).float()
        if audio_chunk.dim() == 1:
            audio_chunk = audio_chunk.unsqueeze(0)
        with torch.no_grad():
            emb = self.classifier.encode_batch(audio_chunk)
        return emb.squeeze().cpu().numpy()

    def verify(self, audio_chunk):
        """Extract embedding and identify speaker."""
        emb = self.extract_embedding(audio_chunk)
        return self.database.identify(emb)


# ─── Diarization Output Handler ─────────────────────────────────────
class DiarizationHandler:
    """
    Receives diarization output from diart and runs verification
    on each detected speaker segment.
    """

    def __init__(self, verifier, audio_source, sample_rate=16000):
        self.verifier = verifier
        self.audio_source = audio_source
        self.sample_rate = sample_rate
        self.speaker_label_map = {}  # diart_label -> (verified_name, last_sim)
        self.active_speakers = set()
        self.last_print_time = 0
        self._audio_buffer = []
        self._buffer_lock = threading.Lock()
        self.chunk_counter = 0

    def on_chunk(self, audio_chunk):
        """Store audio chunks for verification."""
        with self._buffer_lock:
            self._audio_buffer.append(audio_chunk.squeeze().numpy())
            # Keep only last 30 seconds
            max_chunks = int(30 * self.sample_rate / len(audio_chunk.squeeze().numpy()))
            if len(self._audio_buffer) > max_chunks:
                self._audio_buffer = self._audio_buffer[-max_chunks:]

    def _get_audio_numpy(self, audio):
        """Extract numpy array from SlidingWindowFeature or tensor."""
        if isinstance(audio, SlidingWindowFeature):
            return audio.data.squeeze()
        elif hasattr(audio, 'numpy'):
            return audio.squeeze().numpy()
        elif isinstance(audio, np.ndarray):
            return audio.squeeze()
        return np.array(audio).squeeze()

    def __call__(self, result):
        """Called by diart with (annotation, audio) tuple."""
        annotation, audio = result

        if annotation is None or len(annotation) == 0:
            return

        self.chunk_counter += 1
        now = time.time()

        # Get all active speakers in this annotation
        current_speakers = set()
        for segment, track, label in annotation.itertracks(yield_label=True):
            current_speakers.add(label)

        # Extract audio for verification
        audio_np = self._get_audio_numpy(audio)

        # For each speaker detected, try to verify
        new_events = []
        for label in current_speakers:
            if label not in self.speaker_label_map:
                # New speaker detected by diart — run verification
                try:
                    name, sim = self.verifier.verify(audio_np)
                    if name:
                        self.speaker_label_map[label] = (name, sim)
                        new_events.append(f"  [MATCH] {label} -> {name} (similarity: {sim:.3f})")
                    else:
                        self.speaker_label_map[label] = (None, sim)
                        new_events.append(f"  [UNKNOWN] {label} (best similarity: {sim:.3f})")
                except Exception as e:
                    new_events.append(f"  [ERROR] verification failed for {label}: {e}")
                    self.speaker_label_map[label] = (None, 0.0)
            elif self.chunk_counter % 10 == 0:
                # Re-verify periodically
                if len(audio_np) >= self.sample_rate:
                    name, sim = self.verifier.verify(audio_np)
                    old_name = self.speaker_label_map[label][0]
                    self.speaker_label_map[label] = (name, sim)
                    if name != old_name:
                        if name:
                            new_events.append(f"  [RE-ID] {label} -> {name} (similarity: {sim:.3f})")
                        else:
                            new_events.append(f"  [RE-ID] {label} -> UNKNOWN (similarity: {sim:.3f})")

        # Log changes
        if current_speakers != self.active_speakers or new_events:
            timestamp = datetime.now().strftime("%H:%M:%S")

            if new_events:
                for event in new_events:
                    print(f"[{timestamp}] {event}")

            # Show active speaker summary if changed
            if current_speakers != self.active_speakers:
                summary = []
                for label in sorted(current_speakers):
                    info = self.speaker_label_map.get(label, (None, 0.0))
                    if info[0]:
                        summary.append(f"{info[0]}({label})")
                    else:
                        summary.append(f"Unknown({label})")

                if summary:
                    print(f"[{timestamp}] Active: {', '.join(summary)}")
                elif self.active_speakers:
                    print(f"[{timestamp}] Silence")

            self.active_speakers = current_speakers


# ─── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="24/7 Speaker Diarization + Verification")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--device-index", type=int, default=None, help="Audio device index")
    parser.add_argument("--no-verify", action="store_true", help="Run diarization only, no verification")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_rate = config["audio"]["sample_rate"]
    device_index = args.device_index or config["audio"]["device_index"]
    compute_device = config["device"]
    diar_cfg = config["diarization"]
    verif_cfg = config["verification"]

    print("=" * 60)
    print("  Voice ID — Real-Time Speaker Diarization + Verification")
    print("=" * 60)

    # Load speaker database
    print("\n[1/4] Loading enrolled speakers...")
    database = SpeakerDatabase(
        enrollment_dir=verif_cfg["enrollment_dir"],
        threshold=verif_cfg["threshold"],
    )

    # Load verification model
    if not args.no_verify:
        print("\n[2/4] Loading verification model (ECAPA-TDNN)...")
        verifier = SpeakerVerifier(database, compute_device)
    else:
        verifier = None

    # Build diart pipeline
    print(f"\n[3/4] Building diarization pipeline...")
    print(f"  Segmentation: {diar_cfg['segmentation_model']}")
    print(f"  Embedding:    {diar_cfg['embedding_model']}")
    print(f"  Device:       {compute_device}")

    segmentation = SegmentationModel.from_pretrained(diar_cfg["segmentation_model"])
    embedding = EmbeddingModel.from_pretrained(diar_cfg["embedding_model"])

    diar_config = SpeakerDiarizationConfig(
        segmentation=segmentation,
        embedding=embedding,
        step=diar_cfg["step"],
        latency=diar_cfg["latency"],
        tau_active=diar_cfg["tau_active"],
        rho_update=diar_cfg["rho_update"],
        delta_new=diar_cfg["delta_new"],
        max_speakers=diar_cfg["max_speakers"],
        device=torch.device(compute_device),
    )
    pipeline = SpeakerDiarization(diar_config)

    # Set up audio source
    print(f"\n[4/4] Starting microphone stream...")
    import sounddevice as sd
    print(f"\nAvailable audio devices:")
    print(sd.query_devices())
    if device_index is not None:
        print(f"\nUsing device index: {device_index}")

    source = MicrophoneAudioSource(block_duration=diar_cfg["step"], device=device_index)

    # Build handler
    handler = DiarizationHandler(
        verifier=verifier,
        audio_source=source,
        sample_rate=sample_rate,
    ) if verifier else None

    # Graceful shutdown
    def signal_handler(sig, frame):
        print("\n[SHUTDOWN] Stopping...", flush=True)
        os._exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 60)
    print("  MONITORING — Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Run streaming inference
    inference = StreamingInference(
        pipeline=pipeline,
        source=source,
        do_plot=False,
        show_progress=False,
        do_profile=False,
    )

    if handler:
        inference.attach_hooks(handler)

    inference()


if __name__ == "__main__":
    main()
