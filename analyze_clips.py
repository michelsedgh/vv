#!/usr/bin/env python3
"""Analyze diagnostic clips against enrolled speaker anchors.

For each clip in diagnostic_clips/:
  1. Raw audio → WeSpeaker embedding (enrollment-style)
  2. Audio → PixIT separation → per-source WeSpeaker embeddings
  3. Cosine distance to every enrolled anchor (both raw and separated)

Run:  python analyze_clips.py
"""
import sys
import numpy as np
import torch
import yaml
import soundfile as sf
from pathlib import Path
from scipy.spatial.distance import cosine, cdist

ROOT = Path(__file__).parent
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())
DEVICE = CFG.get("device", "cuda")
SR = CFG["audio"]["sample_rate"]
CHUNK = int(CFG["pixit"]["duration"] * SR)  # 80000

# ── Load enrolled anchors ────────────────────────────────────
enroll_dir = ROOT / CFG["enrollment"]["dir"]
anchors_raw = {}       # name → raw embedding (from embedding.npy)
for spk_dir in sorted(enroll_dir.iterdir()):
    emb_file = spk_dir / "embedding.npy"
    if emb_file.exists():
        e = np.load(emb_file).astype(np.float64).ravel()
        e /= (np.linalg.norm(e) + 1e-12)
        anchors_raw[spk_dir.name] = e

if not anchors_raw:
    print("No enrolled speakers found."); sys.exit(1)
print(f"Enrolled speakers: {list(anchors_raw.keys())}")

# ── Load models ──────────────────────────────────────────────
print("Loading PixIT...")
from pixit_wrapper import make_pixit_segmentation_model
seg_model = make_pixit_segmentation_model(CFG["pixit"]["model"])
seg_model.to(torch.device(DEVICE))
pixit = seg_model.model

print("Loading WeSpeaker...")
from pyannote.audio import Model as PyanModel
emb_model = PyanModel.from_pretrained(CFG["embedding"]["model"]).to(torch.device(DEVICE))
emb_model.eval()

def extract_raw_emb(audio):
    """Embedding from raw audio (enrollment-style)."""
    a = np.asarray(audio, dtype=np.float32).flatten()
    peak = np.max(np.abs(a))
    if peak > 1e-6:
        a = a / peak
    wav = torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        e = emb_model(wav)
    e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
    return e.squeeze(0).cpu().numpy().astype(np.float64)

def extract_separated_embs(audio):
    """Run PixIT, return list of (source_idx, activation, embedding)."""
    a = np.asarray(audio, dtype=np.float32).flatten()
    if len(a) < CHUNK:
        a = np.pad(a, (0, CHUNK - len(a)))
    a = a[:CHUNK]

    wav_gpu = torch.from_numpy(a).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    with torch.inference_mode(), torch.cuda.amp.autocast():
        seg_t = pixit(wav_gpu)
    seg_np = seg_t[0].float().cpu().numpy()
    sources = pixit.last_sources.float()

    results = []
    for i in range(seg_np.shape[1]):
        act_max = float(np.max(seg_np[:, i]))
        act_mean = float(np.mean(seg_np[:, i]))
        src = sources[0, :, i]
        peak = src.abs().max().item()
        if peak < 1e-4:
            results.append((i, act_max, act_mean, None))
            continue
        src = src / peak
        inp = src.unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            e = emb_model(inp)
        e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
        emb = e.squeeze(0).cpu().numpy().astype(np.float64)
        results.append((i, act_max, act_mean, emb))
    return results

# ── Also extract enrolled anchors through PixIT ──────────────
print("\n" + "=" * 70)
print("ENROLLED ANCHOR ANALYSIS (PixIT-separated vs raw)")
print("=" * 70)
anchors_sep = {}
for name, raw_emb in anchors_raw.items():
    ref_wav = enroll_dir / name / "reference.wav"
    if not ref_wav.exists():
        print(f"  {name}: no reference.wav, using raw only")
        anchors_sep[name] = raw_emb
        continue
    audio, sr = sf.read(str(ref_wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SR:
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio).unsqueeze(0), sr, SR).squeeze(0).numpy()

    sep_results = extract_separated_embs(audio)
    raw_e = extract_raw_emb(audio)

    print(f"\n  {name} reference.wav:")
    print(f"    Raw emb → stored anchor:  dist={cosine(raw_e, raw_emb):.4f}")
    best_sep_emb = None
    best_sep_act = -1
    for idx, act_max, act_mean, emb in sep_results:
        if emb is None:
            print(f"    Source {idx}: max={act_max:.3f} mean={act_mean:.3f}  [silent]")
            continue
        d = cosine(emb, raw_emb)
        print(f"    Source {idx}: max={act_max:.3f} mean={act_mean:.3f}  → stored anchor dist={d:.4f}")
        if act_mean > best_sep_act:
            best_sep_act = act_mean
            best_sep_emb = emb
    if best_sep_emb is not None:
        anchors_sep[name] = best_sep_emb
        print(f"    → Using source with mean_act={best_sep_act:.3f} as separated anchor")
    else:
        anchors_sep[name] = raw_emb
        print(f"    → All sources silent, using raw anchor")

# ── Analyze diagnostic clips ─────────────────────────────────
diag_dir = ROOT / "diagnostic_clips"
if not diag_dir.exists() or not list(diag_dir.glob("*.wav")):
    print("\nNo diagnostic clips found in diagnostic_clips/")
    print("Use the dashboard to record some clips first.")
    sys.exit(0)

clips = sorted(diag_dir.glob("*.wav"))
print(f"\n{'=' * 70}")
print(f"DIAGNOSTIC CLIPS ({len(clips)} files)")
print(f"{'=' * 70}")

anchor_names = sorted(anchors_raw.keys())

for clip_path in clips:
    audio, sr = sf.read(str(clip_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != SR:
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio).unsqueeze(0), sr, SR).squeeze(0).numpy()

    peak = np.max(np.abs(audio))
    dur = len(audio) / SR

    print(f"\n{'─' * 70}")
    print(f"CLIP: {clip_path.name}  ({dur:.1f}s, peak={peak:.3f})")
    print(f"{'─' * 70}")

    # Raw embedding
    raw_emb = extract_raw_emb(audio)
    print(f"\n  RAW (no separation):")
    for name in anchor_names:
        d_raw = cosine(raw_emb, anchors_raw[name])
        d_sep = cosine(raw_emb, anchors_sep[name])
        print(f"    → {name:>10s}: dist_to_raw_anchor={d_raw:.4f}  dist_to_sep_anchor={d_sep:.4f}")

    # PixIT separated
    sep_results = extract_separated_embs(audio)
    print(f"\n  SEPARATED (PixIT → per-source):")
    for idx, act_max, act_mean, emb in sep_results:
        active = act_max >= CFG["clustering"]["tau_active"]
        tag = "ACTIVE" if active else "inactive"
        if emb is None:
            print(f"    Source {idx} [{tag}]: max={act_max:.3f} mean={act_mean:.3f}  [silent]")
            continue
        print(f"    Source {idx} [{tag}]: max={act_max:.3f} mean={act_mean:.3f}")
        for name in anchor_names:
            d_raw = cosine(emb, anchors_raw[name])
            d_sep = cosine(emb, anchors_sep[name])
            match_raw = "MATCH" if d_raw < CFG["clustering"]["delta_enrolled"] else "no"
            match_sep = "MATCH" if d_sep < CFG["clustering"]["delta_enrolled"] else "no"
            print(f"      → {name:>10s}: vs_raw_anchor={d_raw:.4f} [{match_raw}]"
                  f"  vs_sep_anchor={d_sep:.4f} [{match_sep}]"
                  f"  (threshold={CFG['clustering']['delta_enrolled']})")

print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print(f"  delta_enrolled = {CFG['clustering']['delta_enrolled']}")
print(f"  delta_new      = {CFG['clustering']['delta_new']}")
print(f"  Anchor types: raw (from clean audio), sep (from PixIT-separated)")
print(f"  If sep_anchor distances are much lower → use separated enrollment (already implemented)")
print(f"  If ALL distances are > delta_enrolled → threshold too tight or voice changed")
