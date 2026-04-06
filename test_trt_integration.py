#!/usr/bin/env python3
"""
Test TRT encoder integration with the full PixIT model.
Run after reboot when GPU memory is available.

Usage:
    python test_trt_integration.py
"""
import gc
import time

import numpy as np
import torch

DEVICE = torch.device("cuda")
CHUNK_SAMPLES = 80000


def benchmark_pytorch_only():
    """Baseline: full PixIT in PyTorch fp16."""
    from pyannote.audio import Model as PyanModel
    from pixit_wrapper import PixITWrapper

    print("Loading PixIT (PyTorch fp16)...")
    raw = PyanModel.from_pretrained("pyannote/separation-ami-1.0")
    wrapper = PixITWrapper(raw).to(DEVICE).eval().half()

    dummy = torch.randn(1, 1, CHUNK_SAMPLES, device=DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(5):
            wrapper(dummy)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = wrapper(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    pt_time = np.mean(times[5:])
    print(f"  PyTorch fp16 PixIT: {pt_time:.1f}ms")
    print(f"  Output shape: diarization={out.shape}, sources={wrapper.last_sources.shape}")

    # Capture reference output for quality check
    with torch.inference_mode():
        ref_out = wrapper(dummy)
    ref_diar = ref_out.cpu().numpy()
    ref_src = wrapper.last_sources.cpu().numpy()

    del wrapper, raw
    gc.collect()
    torch.cuda.empty_cache()
    return pt_time, ref_diar, ref_src


def benchmark_trt_hybrid():
    """Hybrid: PixIT with TRT encoder for WavLM."""
    from pyannote.audio import Model as PyanModel
    from pixit_wrapper import PixITWrapper
    from trt_wavlm import patch_wavlm_with_trt, trt_engine_available

    if not trt_engine_available():
        print("  TRT engine not found — skipping.")
        return None, None, None

    print("Loading PixIT (PyTorch fp16 + TRT encoder)...")
    raw = PyanModel.from_pretrained("pyannote/separation-ami-1.0")
    wrapper = PixITWrapper(raw).to(DEVICE).eval().half()

    # Apply TRT patch
    wavlm = wrapper.totatonet.wavlm
    ok = patch_wavlm_with_trt(wavlm, DEVICE)
    if not ok:
        print("  TRT patch failed — skipping.")
        del wrapper, raw
        gc.collect()
        torch.cuda.empty_cache()
        return None, None, None

    dummy = torch.randn(1, 1, CHUNK_SAMPLES, device=DEVICE, dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(5):
            wrapper(dummy)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = wrapper(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    trt_time = np.mean(times[5:])
    print(f"  TRT hybrid PixIT: {trt_time:.1f}ms")
    print(f"  Output shape: diarization={out.shape}, sources={wrapper.last_sources.shape}")

    with torch.inference_mode():
        trt_out = wrapper(dummy)
    trt_diar = trt_out.cpu().numpy()
    trt_src = wrapper.last_sources.cpu().numpy()

    del wrapper, raw
    gc.collect()
    torch.cuda.empty_cache()
    return trt_time, trt_diar, trt_src


def main():
    torch.backends.cudnn.benchmark = True
    props = torch.cuda.get_device_properties(0)
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    total = props.total_memory / 1024**2
    print(f"GPU: {props.name}  |  {alloc:.0f}/{total:.0f} MB")

    # PyTorch baseline
    print("\n=== PyTorch fp16 baseline ===")
    pt_time, ref_diar, ref_src = benchmark_pytorch_only()

    # TRT hybrid
    print("\n=== TRT hybrid ===")
    trt_time, trt_diar, trt_src = benchmark_trt_hybrid()

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"PyTorch fp16:  {pt_time:.1f}ms")
    if trt_time is not None:
        print(f"TRT hybrid:    {trt_time:.1f}ms")
        print(f"Speedup:       {pt_time / trt_time:.2f}x")

        # Quality comparison
        if ref_diar is not None and trt_diar is not None:
            diar_diff = np.max(np.abs(ref_diar - trt_diar))
            src_diff = np.max(np.abs(ref_src - trt_src))
            print(f"Max diarization diff: {diar_diff:.6f}")
            print(f"Max sources diff:     {src_diff:.6f}")
            if diar_diff < 0.01:
                print("Quality: EXCELLENT (< 0.01 max diff)")
            elif diar_diff < 0.1:
                print("Quality: GOOD (< 0.1 max diff)")
            else:
                print("Quality: CHECK NEEDED (> 0.1 max diff)")

        # Pipeline estimate
        # Full pipeline = PixIT + WeSpeaker (~30ms) + clustering (~5ms) + overhead (~15ms)
        est_pipeline = trt_time + 50  # WeSpeaker + rest
        est_baseline = pt_time + 50
        print(f"\nEstimated full pipeline:")
        print(f"  Current (PyTorch fp16): ~{est_baseline:.0f}ms")
        print(f"  With TRT encoder:       ~{est_pipeline:.0f}ms")
    else:
        print("TRT hybrid: not available")


if __name__ == "__main__":
    main()
