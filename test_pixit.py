"""Test PixIT (ToTaToNet) separation-ami-1.0 model on Jetson Orin Nano."""

import torch
import time
import sys
import os

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        return allocated, reserved
    return 0, 0

def main():
    print("=" * 60)
    print("PixIT (ToTaToNet) Benchmark on Jetson Orin Nano")
    print("=" * 60)

    # System info
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"Total GPU memory: {total_mem:.0f} MB")

    alloc, res = get_gpu_memory()
    print(f"GPU memory before loading: {alloc:.0f} MB allocated, {res:.0f} MB reserved")

    # Step 1: Load the model
    print("\n--- Loading pyannote/separation-ami-1.0 ---")
    from pyannote.audio import Model

    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    print(f"HF token: {token[:10]}...")

    t0 = time.time()
    try:
        model = Model.from_pretrained(
            "pyannote/separation-ami-1.0",
            use_auth_token=token,
        )
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s (CPU)")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if model is None:
        print("ERROR: Model is None. You likely need to accept the model conditions.")
        print("Visit: https://hf.co/pyannote/separation-ami-1.0")
        sys.exit(1)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Parameters: {num_params_m:.1f}M")
    print(f"Model size (FP32): {model_size_mb:.0f} MB")

    alloc, res = get_gpu_memory()
    print(f"GPU memory after CPU load: {alloc:.0f} MB allocated, {res:.0f} MB reserved")

    # Step 2: Move to GPU
    print("\n--- Moving to GPU ---")
    t0 = time.time()
    model = model.to(torch.device("cuda"))
    model.eval()
    move_time = time.time() - t0
    print(f"Moved to GPU in {move_time:.1f}s")

    alloc, res = get_gpu_memory()
    print(f"GPU memory after model.to(cuda): {alloc:.0f} MB allocated, {res:.0f} MB reserved")

    # Step 3: Warmup inference
    print("\n--- Warmup inference (first run is slower) ---")
    duration = 5.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    waveform = torch.randn(1, 1, num_samples).to("cuda")

    with torch.inference_mode():
        t0 = time.time()
        output = model(waveform)
        torch.cuda.synchronize()
        warmup_time = time.time() - t0

    # Check output format
    if isinstance(output, tuple):
        print(f"Output is a tuple with {len(output)} elements")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  output[{i}]: shape={o.shape}, dtype={o.dtype}")
            else:
                print(f"  output[{i}]: type={type(o)}")
        diarization = output[0]
        sources = output[1]
    else:
        print(f"Output type: {type(output)}, shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        diarization = output
        sources = None

    print(f"Warmup inference: {warmup_time:.3f}s")

    alloc, res = get_gpu_memory()
    print(f"GPU memory after inference: {alloc:.0f} MB allocated, {res:.0f} MB reserved")

    # Step 4: Benchmark multiple runs
    print("\n--- Benchmarking (10 runs) ---")
    times = []
    with torch.inference_mode():
        for i in range(10):
            waveform = torch.randn(1, 1, num_samples).to("cuda")
            t0 = time.time()
            output = model(waveform)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    rtf = avg_time / duration  # real-time factor

    print(f"\n--- Results ---")
    print(f"Avg inference: {avg_time:.3f}s for {duration}s audio")
    print(f"Min: {min_time:.3f}s, Max: {max_time:.3f}s")
    print(f"Real-time factor: {rtf:.3f}x (need < 1.0 for real-time)")
    print(f"Per 500ms step (diart default): ~{avg_time:.3f}s (need < 0.5s)")

    if rtf < 0.1:
        print("VERDICT: Excellent - easily real-time, plenty of headroom")
    elif rtf < 0.5:
        print("VERDICT: Good - real-time capable with diart's 500ms step")
    elif rtf < 1.0:
        print("VERDICT: Marginal - processes faster than real-time but tight for diart")
    else:
        print("VERDICT: TOO SLOW - cannot keep up with real-time audio")

    # Step 5: Test FP16
    print("\n--- Testing FP16 ---")
    model_fp16 = model.half()
    waveform_fp16 = torch.randn(1, 1, num_samples, dtype=torch.float16).to("cuda")

    try:
        with torch.inference_mode():
            t0 = time.time()
            output_fp16 = model_fp16(waveform_fp16)
            torch.cuda.synchronize()
            fp16_time = time.time() - t0
            print(f"FP16 inference: {fp16_time:.3f}s (vs FP32 avg: {avg_time:.3f}s)")
            alloc, res = get_gpu_memory()
            print(f"GPU memory (FP16): {alloc:.0f} MB allocated, {res:.0f} MB reserved")
    except Exception as e:
        print(f"FP16 failed: {e}")

    # Step 6: Analyze outputs
    print("\n--- Output Analysis ---")
    if sources is not None:
        print(f"Diarization: shape={diarization.shape}")
        print(f"  -> (batch={diarization.shape[0]}, frames={diarization.shape[1]}, speakers={diarization.shape[2]})")
        print(f"  -> values range: [{diarization.min():.3f}, {diarization.max():.3f}]")
        print(f"Sources: shape={sources.shape}")
        print(f"  -> (batch={sources.shape[0]}, samples={sources.shape[1]}, speakers={sources.shape[2]})")
        print(f"  -> values range: [{sources.min():.3f}, {sources.max():.3f}]")
    else:
        print("No separated sources in output")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()
