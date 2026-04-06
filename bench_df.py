#!/usr/bin/env python3
"""Benchmark DeepFilterNet: GPU-transfer path vs CPU-only path."""
import time
import numpy as np
import torch
import torchaudio.functional as AF
from df import enhance, init_df

SR = 16000
CHUNK = 80000  # 5s
N_ITER = 30

print("Loading DeepFilterNet3...")
df_model, df_state, _ = init_df(model_base_dir="DeepFilterNet3")
df_sr = df_state.sr()  # 48000
print(f"  Native SR: {df_sr}")

rng = np.random.default_rng(42)
test_wav = rng.standard_normal(CHUNK).astype(np.float32) * 0.3

# Warmup
wav_t = torch.from_numpy(test_wav).unsqueeze(0)
wav_48 = AF.resample(wav_t, SR, df_sr)
_ = enhance(df_model, df_state, wav_48)

# ── A) Current path: numpy→GPU→resample→CPU→enhance→GPU→resample→CPU ──
print(f"\n=== Path A: GPU resample + CPU enhance ({N_ITER} iters) ===")
device = torch.device("cuda")
times_a = {"to_gpu": [], "resample_up": [], "to_cpu": [], "enhance": [],
           "to_gpu2": [], "resample_down": [], "to_cpu2": [], "total": []}

for i in range(N_ITER):
    waveform = test_wav.copy()
    torch.cuda.synchronize()
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    wav_t = torch.from_numpy(waveform).to(device).unsqueeze(0)
    torch.cuda.synchronize()
    times_a["to_gpu"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    wav_48 = AF.resample(wav_t, SR, df_sr)
    torch.cuda.synchronize()
    times_a["resample_up"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    wav_48_cpu = wav_48.cpu()
    times_a["to_cpu"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    wav_48_cpu = enhance(df_model, df_state, wav_48_cpu)
    times_a["enhance"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    wav_16_gpu = AF.resample(wav_48_cpu.to(device), df_sr, SR)
    torch.cuda.synchronize()
    times_a["to_gpu2"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    _ = wav_16_gpu.squeeze(0).cpu().numpy()[:CHUNK]
    times_a["to_cpu2"].append(time.perf_counter() - t0)

    times_a["total"].append(time.perf_counter() - t_total)

print(f"{'Component':<18s} {'Mean':>7s} {'Min':>7s} {'Max':>7s} ms")
print("-" * 45)
for name, vals in times_a.items():
    arr = np.array(vals[3:]) * 1000
    print(f"{name:<18s} {arr.mean():7.1f} {arr.min():7.1f} {arr.max():7.1f}")

# ── B) CPU-only path ──
print(f"\n=== Path B: All CPU ({N_ITER} iters) ===")
times_b = {"resample_up": [], "enhance": [], "resample_down": [], "total": []}

for i in range(N_ITER):
    waveform = test_wav.copy()
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    wav_t = torch.from_numpy(waveform).unsqueeze(0)  # CPU
    wav_48 = AF.resample(wav_t, SR, df_sr)           # CPU
    times_b["resample_up"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    wav_48 = enhance(df_model, df_state, wav_48)
    times_b["enhance"].append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    wav_16 = AF.resample(wav_48, df_sr, SR)           # CPU
    _ = wav_16.squeeze(0).numpy()[:CHUNK]
    times_b["resample_down"].append(time.perf_counter() - t0)

    times_b["total"].append(time.perf_counter() - t_total)

print(f"{'Component':<18s} {'Mean':>7s} {'Min':>7s} {'Max':>7s} ms")
print("-" * 45)
for name, vals in times_b.items():
    arr = np.array(vals[3:]) * 1000
    print(f"{name:<18s} {arr.mean():7.1f} {arr.min():7.1f} {arr.max():7.1f}")

# ── C) CPU-only with pre-computed resample kernel ──
print(f"\n=== Path C: CPU + pre-computed kernel ({N_ITER} iters) ===")
# Pre-compute the resample kernel once
up_kernel = AF._get_sinc_resample_kernel(SR, df_sr, 6, 64, dtype=torch.float32)  
# Try torchaudio's Resample transform which caches kernels
from torchaudio.transforms import Resample
resample_up = Resample(SR, df_sr)
resample_down = Resample(df_sr, SR)

# warmup
_ = resample_up(torch.from_numpy(test_wav).unsqueeze(0))

times_c = {"total": []}
for i in range(N_ITER):
    waveform = test_wav.copy()
    t0 = time.perf_counter()
    wav_t = torch.from_numpy(waveform).unsqueeze(0)
    wav_48 = resample_up(wav_t)
    wav_48 = enhance(df_model, df_state, wav_48)
    wav_16 = resample_down(wav_48)
    _ = wav_16.squeeze(0).numpy()[:CHUNK]
    times_c["total"].append(time.perf_counter() - t0)

arr = np.array(times_c["total"][3:]) * 1000
print(f"  Total:  mean={arr.mean():.1f}  min={arr.min():.1f}  max={arr.max():.1f} ms")
