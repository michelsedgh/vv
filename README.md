# Voice — MeanFlow-TSE live mic (STT-oriented)

This repo includes a **web dashboard + API** that runs **[MeanFlow-TSE](https://github.com/rikishimizu/MeanFlow-TSE)** on the microphone: enroll a short reference, then stream **extracted target speech** (16 kHz) over WebSockets for playback or an STT backend.

### Quick start (MeanFlow)

**Orin Nano / Jetson:** use **`requirements_orin_meanflow.txt` + constraints** so pip never swaps your **NVIDIA torch 2.3.x** for a random PyPI `torch` wheel.

Do **not** run `pip install -r meanflow_tse/requirements.txt` on Jetson — that file is for **x86 + cu118** only (see warning at top of that file).

```bash
cd ~/Documents/Voice
conda activate voice_id

# 1) Install PyTorch the NVIDIA way (torch 2.3.0 + matching torchaudio for your JetPack):
#    https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/

# 2) If an earlier `pip install` broke torch, reinstall your `.whl` files first.

# 3) Freeze torch* versions, then install MeanFlow deps WITHOUT upgrading torch:
chmod +x scripts/mk_constraints_orin.sh
./scripts/mk_constraints_orin.sh
pip install -r requirements_orin_meanflow.txt -c constraints_orin.txt

# (Optional) If you have no torch yet, copy and edit versions:
#   cp constraints_orin.example.txt constraints_orin.txt

# Download MeanFlow + t-predictor weights → models/meanflow/ (see models/meanflow/README.md).
# config.yaml: 16 kHz, enroll/record 3 s (segment_aux), chunk 3 s, use_t_predictor + noisy ckpts.

python server.py
# Open http://<orin-ip>:8042 — Enroll, then Start.
```

**x86 Linux + CUDA 11.8 (not Jetson):** use `meanflow_tse/requirements.txt` (includes torch 2.0.1 + cu118 index).

- **Config:** `config.yaml` — `audio.sample_rate` 16000, `enrollment.record_seconds`, `meanflow.enroll_seconds` / `chunk_seconds` (3), `use_t_predictor`, checkpoint paths.
- **Mic overflow:** increase mic `block` in `server.py` or reduce GPU load; streaming uses FIFO `chunk_seconds` segments (see `max_queued_chunks` in `config.yaml`).

### Git graph looks empty in Cursor / VS Code?

This repo often has **only a couple of commits** on `main` and **no `origin` remote** until you add one. The branch graph then shows almost nothing. Add your hosting remote and push, e.g. `git remote add origin <url>` then `git push -u origin main`, and the graph will populate as you commit and fetch.

---

# Voice ID — Real-Time Speaker Diarization + Verification (legacy)

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
