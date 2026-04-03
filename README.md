# Voice ID — Real-Time Speaker Diarization + Verification

Real-time speaker diarization and verification system for Jetson Orin Nano 8GB.
Continuously monitors a microphone, identifies "who spoke when" via diart, and verifies
each detected speaker against pre-enrolled household voice profiles.

## Stack

| Component | Library | Model |
|---|---|---|
| Segmentation | pyannote.audio | `pyannote/segmentation` (1.5M params) |
| Embedding | SpeechBrain | `speechbrain/spkrec-ecapa-voxceleb` (ECAPA-TDNN, 192-dim) |
| Diarization | diart | Real-time streaming pipeline |
| Verification | Custom | Cosine similarity against enrolled embeddings |
| Hardware | Jetson Orin Nano 8GB | PyTorch 2.3.0 + CUDA 12.4 |

## Setup

### Prerequisites
- Jetson Orin Nano with JetPack R36.x
- Conda (miniconda3)
- USB microphone
- HuggingFace account with token (for pyannote gated model)

### Environment (already created)

```bash
conda activate voice_id
```

If recreating from scratch:
```bash
conda create -n voice_id python=3.10 -y
conda activate voice_id

# Install Jetson PyTorch wheel (CUDA 12.4)
pip install ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip install ~/Downloads/torchvision-0.18.0-cp310-cp310-linux_aarch64.whl
pip install ~/Downloads/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl

# Install diart + dependencies
pip install 'numpy<2.0.0' diart sounddevice

# System deps
sudo apt-get install -y portaudio19-dev libportaudio2

# HuggingFace login (needed for pyannote/segmentation gated model)
# First accept terms at: https://huggingface.co/pyannote/segmentation
huggingface-cli login
```

## Usage

### 1. Enroll Speakers

Record 3 samples per speaker (10 seconds each by default):

```bash
conda activate voice_id
cd ~/Documents/Voice

# Interactive enrollment (records from mic)
python enroll_speaker.py --name "Michel" --duration 10 --samples 3

# Or from existing WAV file
python enroll_speaker.py --name "Michel" --wav /path/to/voice.wav
```

Repeat for each household member. Embeddings are saved in `enrolled_speakers/<name>/`.

### 2. Run 24/7 Monitor

```bash
python stream_monitor.py
```

Options:
- `--config config.yaml` — custom config file
- `--device-index 0` — override audio device
- `--no-verify` — diarization only, no speaker verification

### 3. Output

The system prints real-time events:
```
[14:30:05]   [MATCH] speaker0 -> Michel (similarity: 0.782)
[14:30:05] Active: Michel(speaker0)
[14:30:12]   [UNKNOWN] speaker1 (best similarity: 0.231)
[14:30:12] Active: Michel(speaker0), Unknown(speaker1)
[14:30:20] Silence
```

## Configuration

Edit `config.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `audio.device_index` | `0` | Microphone device index |
| `diarization.step` | `0.5` | Seconds between chunk shifts |
| `diarization.latency` | `5.0` | Streaming latency (lower = faster but less accurate) |
| `diarization.tau_active` | `0.5` | Speech activity threshold |
| `diarization.delta_new` | `1.0` | New speaker detection threshold |
| `diarization.max_speakers` | `6` | Max concurrent speakers |
| `verification.threshold` | `0.45` | Cosine similarity threshold for match |
| `device` | `cuda` | `cuda` or `cpu` |

## Tuning Tips

- **Verification threshold**: Start at 0.45, increase to reduce false positives, decrease to reduce false negatives. Test with your actual speakers.
- **Enrollment quality**: Record in the same environment the system will run in. Multiple samples improve robustness.
- **Latency**: 5s gives best accuracy. Can reduce to 2-3s for faster response at slight accuracy cost.
- **tau_active**: Lower values detect quieter speech, higher values reduce false activations.

## Architecture

```
Microphone (16kHz mono)
    │
    ▼
┌─────────────────────────────────┐
│  diart StreamingInference       │
│  ┌───────────────────────────┐  │
│  │ pyannote/segmentation     │  │  ← VAD + overlap detection
│  │ (who is speaking when)    │  │
│  └───────────┬───────────────┘  │
│              │                  │
│  ┌───────────▼───────────────┐  │
│  │ ECAPA-TDNN embeddings     │  │  ← 192-dim speaker vectors
│  │ + incremental clustering  │  │
│  └───────────┬───────────────┘  │
│              │                  │
│  Output: speaker-labeled        │
│  segments with timestamps       │
└──────────────┬──────────────────┘
               │
    ┌──────────▼──────────┐
    │ Verification Layer  │
    │ ECAPA-TDNN embedding│  ← Extract embedding from segment
    │ vs enrolled profiles│  ← Cosine similarity comparison
    └──────────┬──────────┘
               │
    Known Speaker / Unknown
```

## Notes

- **Why ECAPA-TDNN instead of TitaNet?** NeMo (required for TitaNet) has unresolved compatibility issues on Jetson ARM64 with PyTorch 2.3. ECAPA-TDNN produces equivalent 192-dim embeddings with comparable accuracy and is lighter/faster.
- The pyannote segmentation model version warnings are harmless — the model works correctly despite being trained on older versions.
- For 24/7 operation, consider running via `systemd` service or `tmux`/`screen` session.
