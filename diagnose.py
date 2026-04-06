#!/usr/bin/env python3
"""Diagnose enrolled speaker matching.

Records 5s of mic audio, runs PixIT + WeSpeaker, and prints cosine distances
between every separated-source embedding and the enrolled anchor.
"""
import sys, time, yaml, numpy as np, torch, sounddevice as sd
from pathlib import Path
from scipy.spatial.distance import cosine

ROOT = Path(__file__).parent
CFG  = yaml.safe_load((ROOT / "config.yaml").read_text())

SR       = CFG["audio"]["sample_rate"]
DEV_IDX  = CFG["audio"]["device_index"]
DURATION = CFG["pixit"]["duration"]
DEVICE   = CFG.get("device", "cuda")

# ── 1. Load enrolled anchor ─────────────────────────────────────
enrolled_dir = ROOT / CFG["enrollment"]["dir"]
anchors = {}
for spk_dir in sorted(enrolled_dir.iterdir()):
    emb_file = spk_dir / "embedding.npy"
    if emb_file.exists():
        emb = np.load(emb_file).astype(np.float64).ravel()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        anchors[spk_dir.name] = emb
        print(f"  Enrolled: {spk_dir.name}  dim={emb.shape[0]}  norm={np.linalg.norm(emb):.4f}")

if not anchors:
    print("ERROR: No enrolled speakers found"); sys.exit(1)

# ── 2. Record 5s of mic audio ───────────────────────────────────
chunk_samples = int(DURATION * SR)
print(f"\n🎤 Recording {DURATION}s of audio... SPEAK NOW!")
audio = sd.rec(chunk_samples, samplerate=SR, channels=1, dtype="float32", device=DEV_IDX)
sd.wait()
audio = audio[:, 0]
peak = np.max(np.abs(audio))
print(f"  Recorded {len(audio)} samples, peak={peak:.4f}")

if peak < 0.01:
    print("WARNING: Audio is very quiet — check mic?")

# ── 3. Load models ──────────────────────────────────────────────
print("\nLoading PixIT...")
from pixit_wrapper import make_pixit_segmentation_model
seg_model = make_pixit_segmentation_model(CFG["pixit"]["model"])
seg_model.to(torch.device(DEVICE))
pixit = seg_model.model  # underlying PixITWrapper

print("Loading WeSpeaker...")
from pyannote.audio import Model as PyanModel
emb_model = PyanModel.from_pretrained(CFG["embedding"]["model"])
emb_model.to(torch.device(DEVICE))
emb_model.eval()

# ── 4. Run PixIT ────────────────────────────────────────────────
wav_gpu = torch.from_numpy(audio).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

with torch.inference_mode(), torch.cuda.amp.autocast():
    diarization = pixit(wav_gpu)

seg_np = diarization[0].float().cpu().numpy()          # (frames, 3)
sources = pixit.last_sources.float()                    # (1, samples, 3) GPU

n_sources = seg_np.shape[1]
print(f"\nPixIT output: {seg_np.shape[0]} frames × {n_sources} sources")

# ── 5. Per-source analysis ──────────────────────────────────────
for src_idx in range(n_sources):
    act_max = np.max(seg_np[:, src_idx])
    act_mean = np.mean(seg_np[:, src_idx])
    print(f"\n{'='*60}")
    print(f"Source {src_idx}: max_act={act_max:.3f}  mean_act={act_mean:.3f}")
    
    active = act_max >= CFG["clustering"]["tau_active"]
    long_spk = act_mean >= CFG["clustering"]["rho_update"]
    print(f"  active (>={CFG['clustering']['tau_active']})? {active}")
    print(f"  long   (>={CFG['clustering']['rho_update']})? {long_spk}")

    if not active:
        print("  SKIPPED (inactive)")
        continue

    # Extract source audio
    src_audio = sources[0, :, src_idx]  # GPU
    src_peak = src_audio.abs().max().item()
    print(f"  src_peak={src_peak:.6f}")
    
    if src_peak < 1e-4:
        print("  SKIPPED (silent source)")
        continue

    # Peak-normalize (same as pipeline)
    src_audio = src_audio / src_peak
    
    # Extract embedding (same as pipeline)
    src_wav = src_audio.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)
    with torch.inference_mode():
        emb_t = emb_model(src_wav)   # (1, dim)
    emb_t = emb_t / (emb_t.norm(dim=-1, keepdim=True) + 1e-8)
    emb_np = emb_t.squeeze(0).cpu().numpy().astype(np.float64)
    
    print(f"  emb_dim={emb_np.shape[0]}  emb_norm={np.linalg.norm(emb_np):.4f}")
    
    # Check for NaN
    if np.isnan(emb_np).any():
        print("  WARNING: NaN embedding!")
        continue
    
    # Distance to each enrolled anchor
    for name, anchor in anchors.items():
        dist = cosine(emb_np, anchor)
        match_enrolled = dist < CFG["clustering"]["delta_enrolled"]
        match_new = dist < CFG["clustering"]["delta_new"]
        status = "MATCH-ENROLLED" if match_enrolled else ("match-new" if match_new else "NO MATCH")
        print(f"  → {name}: cosine_dist={dist:.4f}  [{status}]"
              f"  (delta_enrolled={CFG['clustering']['delta_enrolled']}"
              f"  delta_new={CFG['clustering']['delta_new']})")

# ── 6. Also test raw audio embedding (no separation) ────────────
print(f"\n{'='*60}")
print("RAW MIX (no separation — direct from mic):")
raw_peak = np.max(np.abs(audio))
if raw_peak > 1e-6:
    raw_norm = audio / raw_peak
else:
    raw_norm = audio
raw_wav = torch.from_numpy(raw_norm).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
with torch.inference_mode():
    raw_emb_t = emb_model(raw_wav)
raw_emb_t = raw_emb_t / (raw_emb_t.norm(dim=-1, keepdim=True) + 1e-8)
raw_emb = raw_emb_t.squeeze(0).cpu().numpy().astype(np.float64)
for name, anchor in anchors.items():
    dist = cosine(raw_emb, anchor)
    print(f"  → {name}: cosine_dist={dist:.4f}")

# ── 7. Also test enrollment embedding extraction path ────────────
print(f"\n{'='*60}")
print("ENROLLMENT-STYLE extraction (raw audio, same as enroll endpoint):")
# This is what pipeline.extract_embedding does
from pipeline import RealtimePipeline
pipeline = RealtimePipeline.__new__(RealtimePipeline)
pipeline.device = torch.device(DEVICE)
pipeline._emb_model = emb_model
enroll_emb = pipeline.extract_embedding(audio)
for name, anchor in anchors.items():
    dist = cosine(enroll_emb, anchor)
    print(f"  → {name}: cosine_dist={dist:.4f}")

print(f"\n{'='*60}")
print("DONE. Check distances above to diagnose matching.")
