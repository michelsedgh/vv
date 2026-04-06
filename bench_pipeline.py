#!/usr/bin/env python3
"""Standalone pipeline profiler — measures each stage independently."""
import time
import numpy as np
import torch
import torchaudio.functional as AF
import yaml
from pyannote.audio import Model as PyanModel
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pixit_wrapper import make_pixit_segmentation_model

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

SR = cfg["audio"]["sample_rate"]
DURATION = cfg["pixit"]["duration"]
CHUNK = int(DURATION * SR)  # 80000
N_SPK = cfg["pixit"]["max_speakers"]  # 3

# ── Load models ──────────────────────────────────────────────────
print("Loading PixIT...")
seg_model = make_pixit_segmentation_model(cfg["pixit"]["model"])
seg_model.to(DEVICE)
seg_model.eval()
pixit = seg_model.model  # PixITWrapper

print("Loading WeSpeaker...")
emb_model = PyanModel.from_pretrained(cfg["embedding"]["model"]).to(DEVICE)
emb_model.eval()

# DeepFilterNet
df_model = df_state = df_enhance = None
df_sr = 48000
dcfg = cfg.get("denoiser", {})
if dcfg.get("enabled", False):
    try:
        from df import enhance, init_df
        print("Loading DeepFilterNet...")
        df_model, df_state, _ = init_df(model_base_dir=dcfg.get("model", "DeepFilterNet3"))
        df_enhance = enhance
        df_sr = df_state.sr()
        print(f"  DeepFilterNet ready ({df_sr} Hz)")
    except Exception as e:
        print(f"  DeepFilterNet failed: {e}")

# ── Warmup ───────────────────────────────────────────────────────
print("Warming up...")
dummy = torch.randn(1, 1, CHUNK, device=DEVICE)
with torch.inference_mode():
    pixit(dummy)
    emb_model(dummy)
    emb_model(dummy.expand(N_SPK, -1, -1))
torch.cuda.synchronize()
print("Ready.\n")

# ── Generate test audio (white noise, simulates active speech) ───
rng = np.random.default_rng(42)
test_wav = rng.standard_normal(CHUNK).astype(np.float32) * 0.3

N_ITER = 20

# ── Benchmark each stage ─────────────────────────────────────────
times = {
    "deepfilter": [],
    "pixit": [],
    "wespeaker_batch": [],
    "sources_to_cpu": [],
    "total_gpu": [],
}

for i in range(N_ITER):
    waveform = test_wav.copy()
    torch.cuda.synchronize()
    t_total = time.perf_counter()

    # 0. DeepFilterNet
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if df_model is not None:
        wav_t = torch.from_numpy(waveform).to(DEVICE).unsqueeze(0)
        wav_48 = AF.resample(wav_t, SR, df_sr)
        wav_48_cpu = wav_48.cpu()
        wav_48_cpu = df_enhance(df_model, df_state, wav_48_cpu)
        wav_16 = AF.resample(wav_48_cpu.to(DEVICE), df_sr, SR)
        waveform = wav_16.squeeze(0).cpu().numpy()[:CHUNK]
    torch.cuda.synchronize()
    times["deepfilter"].append(time.perf_counter() - t0)

    # 1. PixIT
    wav_gpu = torch.from_numpy(waveform).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        diarization_t = pixit(wav_gpu)
    torch.cuda.synchronize()
    times["pixit"].append(time.perf_counter() - t0)

    seg_np = diarization_t[0].cpu().numpy()
    sources = pixit.last_sources  # (1, 80000, 3) GPU

    # 2. Batch WeSpeaker
    active_mask = np.max(seg_np, axis=0) >= 0.5
    batch_tensors = []
    for spk_idx in range(N_SPK):
        if not active_mask[spk_idx]:
            continue
        src = sources[0, :, spk_idx]
        peak = src.abs().max()
        if peak < 1e-4:
            continue
        batch_tensors.append((src / peak).unsqueeze(0).unsqueeze(0))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if batch_tensors:
        batch = torch.cat(batch_tensors, dim=0)
        with torch.inference_mode():
            embs = emb_model(batch)
        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-8)
        _ = embs.cpu()
    torch.cuda.synchronize()
    times["wespeaker_batch"].append(time.perf_counter() - t0)

    # 3. Sources to CPU
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = sources[0].cpu().numpy()
    torch.cuda.synchronize()
    times["sources_to_cpu"].append(time.perf_counter() - t0)

    torch.cuda.synchronize()
    times["total_gpu"].append(time.perf_counter() - t_total)

# ── Report ────────────────────────────────────────────────────────
print(f"\n{'Stage':<22s} {'Mean':>8s} {'Min':>8s} {'Max':>8s}  (ms, {N_ITER} iters)")
print("-" * 58)
for name, vals in times.items():
    arr = np.array(vals[2:]) * 1000  # skip first 2 warmup iters
    print(f"{name:<22s} {arr.mean():8.1f} {arr.min():8.1f} {arr.max():8.1f}")

# ── Benchmark PixIT in float16 ────────────────────────────────────
print("\n\n=== PixIT float16 (autocast) ===")
# warmup fp16
wav_gpu = torch.randn(1, 1, CHUNK, device=DEVICE)
with torch.inference_mode(), torch.cuda.amp.autocast():
    for _ in range(3):
        pixit(wav_gpu)
torch.cuda.synchronize()

fp16_times = []
for i in range(N_ITER):
    wav_gpu = torch.from_numpy(test_wav.copy()).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast():
        diarization_t = pixit(wav_gpu)
    torch.cuda.synchronize()
    fp16_times.append(time.perf_counter() - t0)

arr = np.array(fp16_times[2:]) * 1000
print(f"  PixIT fp16:  mean={arr.mean():.1f}  min={arr.min():.1f}  max={arr.max():.1f} ms")

# ── Benchmark WeSpeaker in float16 ───────────────────────────────
print("\n=== WeSpeaker float16 (autocast) ===")
sources = pixit.last_sources
batch_t = []
for spk_idx in range(N_SPK):
    src = sources[0, :, spk_idx]
    peak = src.abs().max()
    if peak > 1e-4:
        batch_t.append((src / peak).unsqueeze(0).unsqueeze(0))
if batch_t:
    batch = torch.cat(batch_t, dim=0)
    # warmup
    with torch.inference_mode(), torch.cuda.amp.autocast():
        for _ in range(3):
            emb_model(batch)
    torch.cuda.synchronize()

    ws_fp16 = []
    for i in range(N_ITER):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.amp.autocast():
            embs = emb_model(batch)
        torch.cuda.synchronize()
        ws_fp16.append(time.perf_counter() - t0)
    arr = np.array(ws_fp16[2:]) * 1000
    print(f"  WeSpeaker fp16 (batch={batch.shape[0]}):  mean={arr.mean():.1f}  min={arr.min():.1f}  max={arr.max():.1f} ms")

# ── Summary ──────────────────────────────────────────────────────
print("\n=== Projected totals ===")
pixit_fp32 = np.mean(np.array(times["pixit"][2:])) * 1000
pixit_fp16_mean = np.mean(np.array(fp16_times[2:])) * 1000
df_mean = np.mean(np.array(times["deepfilter"][2:])) * 1000
ws_mean = np.mean(np.array(times["wespeaker_batch"][2:])) * 1000
cpu_mean = np.mean(np.array(times["sources_to_cpu"][2:])) * 1000

print(f"  Current (fp32 + DF):      {pixit_fp32 + df_mean + ws_mean + cpu_mean:.0f} ms")
print(f"  fp16 PixIT + DF:          {pixit_fp16_mean + df_mean + ws_mean + cpu_mean:.0f} ms")
print(f"  fp32 PixIT, no DF:        {pixit_fp32 + ws_mean + cpu_mean:.0f} ms")
print(f"  fp16 PixIT, no DF:        {pixit_fp16_mean + ws_mean + cpu_mean:.0f} ms")
