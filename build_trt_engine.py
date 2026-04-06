#!/usr/bin/env python3
"""
Build TensorRT engines for WavLM acceleration.
Run this after a FRESH REBOOT for maximum available memory.

Usage:
    python build_trt_engine.py          # runs all steps
    python build_trt_engine.py export   # step 1 only: ONNX export
    python build_trt_engine.py convert  # step 2 only: fp16 ONNX conversion
    python build_trt_engine.py build    # step 3 only: TRT engine build
    python build_trt_engine.py bench    # step 4 only: benchmark

Each step runs as a SEPARATE PROCESS to maximise available GPU memory.
"""
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np

SCRIPT = os.path.abspath(__file__)
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "engines")
ONNX_FP32 = os.path.join(ENGINES_DIR, "wavlm_full_fp32.onnx")
ONNX_FP16 = os.path.join(ENGINES_DIR, "wavlm_full_fp16.onnx")
ONNX_ENC_FP16 = os.path.join(ENGINES_DIR, "wavlm_encoder_fp16.onnx")
TRT_FULL_FP16 = os.path.join(ENGINES_DIR, "wavlm_full_fp16.engine")
TRT_ENC_FP16 = os.path.join(ENGINES_DIR, "wavlm_encoder_fp16.engine")
TRT_CACHE = os.path.join(ENGINES_DIR, "trt_cache")
RESULTS_FILE = os.path.join(ENGINES_DIR, "benchmark_results.json")
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"
PYTHON = sys.executable

CHUNK_SAMPLES = 80000  # 5s at 16kHz
SEQ_LEN = 249          # WavLM output frames for 80000 samples
HIDDEN = 1024          # WavLM-Large hidden size


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}", flush=True)


def gpu_info():
    import torch
    props = torch.cuda.get_device_properties(0)
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    total = props.total_memory / 1024**2
    free = total - alloc
    print(f"GPU: {props.name}  |  {alloc:.0f}/{total:.0f} MB used  |  ~{free:.0f} MB free")


def run_step(step_name):
    """Run a single step in a FRESH subprocess to free all GPU memory."""
    print(f"\n>>> Launching subprocess: {step_name}")
    r = subprocess.run(
        [PYTHON, SCRIPT, step_name],
        cwd=os.path.dirname(SCRIPT),
    )
    return r.returncode == 0


# ──────────────────────────────────────────────────────────────
# Step 1: Export ONNX (fp32)
# ──────────────────────────────────────────────────────────────
def do_export():
    import torch
    banner("Step 1: Export WavLM to ONNX (fp32)")
    os.makedirs(ENGINES_DIR, exist_ok=True)
    gpu_info()

    if os.path.exists(ONNX_FP32):
        sz = os.path.getsize(ONNX_FP32) / 1024**2
        print(f"[skip] Already exists: {ONNX_FP32} ({sz:.0f} MB)")
        return

    from pyannote.audio import Model as PyanModel
    model = PyanModel.from_pretrained("pyannote/separation-ami-1.0")
    model.to("cuda").eval()
    wavlm = model.wavlm

    dummy = torch.randn(1, CHUNK_SAMPLES, device="cuda")
    print("Exporting full WavLM (opset 18)...")
    with torch.inference_mode():
        torch.onnx.export(
            wavlm, dummy, ONNX_FP32,
            input_names=["audio"],
            output_names=["hidden_states"],
            opset_version=18,
            do_constant_folding=True,
        )
    sz = os.path.getsize(ONNX_FP32) / 1024**2
    print(f"  -> {ONNX_FP32} ({sz:.0f} MB)")

    # Also export encoder-only
    encoder = wavlm.encoder
    with torch.inference_mode():
        feats = wavlm.feature_extractor(dummy)
        feats = feats.transpose(1, 2)
        feats, _ = wavlm.feature_projection(feats)

    class EncWrap(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
        def forward(self, x):
            o = self.enc(x)
            return o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]

    wrap = EncWrap(encoder).to("cuda").eval()
    enc_fp32 = os.path.join(ENGINES_DIR, "wavlm_encoder_fp32.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            wrap, feats, enc_fp32,
            input_names=["features"],
            output_names=["hidden_states"],
            opset_version=18,
            do_constant_folding=True,
        )
    sz2 = os.path.getsize(enc_fp32) / 1024**2
    print(f"  -> {enc_fp32} ({sz2:.0f} MB)")

    # PyTorch fp16 baseline
    print("\nBenchmarking PyTorch WavLM fp16...")
    wavlm.half()
    d = torch.randn(1, CHUNK_SAMPLES, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        for _ in range(5):
            wavlm(d)
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            wavlm(d)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    pt = np.mean(times[5:])
    print(f"  PyTorch fp16: {pt:.1f}ms")

    os.makedirs(ENGINES_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump({"pytorch_fp16_ms": round(pt, 1)}, f)

    print("Done. PyTorch model released.")


# ──────────────────────────────────────────────────────────────
# Step 2: Convert ONNX fp32 → fp16  (CPU only, no GPU needed)
# ──────────────────────────────────────────────────────────────
def do_convert():
    banner("Step 2: Convert ONNX fp32 → fp16 (halves memory)")

    import onnx
    from onnxconverter_common import float16

    for src, dst, label in [
        (ONNX_FP32, ONNX_FP16, "full WavLM"),
        (os.path.join(ENGINES_DIR, "wavlm_encoder_fp32.onnx"), ONNX_ENC_FP16, "encoder-only"),
    ]:
        if os.path.exists(dst):
            sz = os.path.getsize(dst) / 1024**2
            print(f"[skip] {label} fp16 already exists: {dst} ({sz:.0f} MB)")
            continue
        if not os.path.exists(src):
            print(f"[skip] {label} fp32 ONNX not found: {src}")
            continue

        sz_in = os.path.getsize(src) / 1024**2
        print(f"Converting {label}: {sz_in:.0f} MB fp32 → fp16...")
        model = onnx.load(src)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, dst)
        sz_out = os.path.getsize(dst) / 1024**2
        print(f"  -> {dst} ({sz_out:.0f} MB)  [{sz_in/sz_out:.1f}x smaller]")
        del model, model_fp16
        gc.collect()

    print("Done.")


# ──────────────────────────────────────────────────────────────
# Step 3: Build TRT engines
# ──────────────────────────────────────────────────────────────
def do_build():
    import torch
    banner("Step 3: Build TensorRT engines from fp16 ONNX")
    gpu_info()

    for onnx_path, engine_path, label in [
        (ONNX_ENC_FP16, TRT_ENC_FP16, "encoder-only fp16"),
        (ONNX_FP16, TRT_FULL_FP16, "full WavLM fp16"),
    ]:
        if os.path.exists(engine_path):
            sz = os.path.getsize(engine_path) / 1024**2
            print(f"\n[skip] {label} engine exists: {engine_path} ({sz:.0f} MB)")
            continue
        if not os.path.exists(onnx_path):
            print(f"\n[skip] {label} ONNX not found: {onnx_path}")
            continue

        sz = os.path.getsize(onnx_path) / 1024**2
        print(f"\nBuilding {label} engine from {sz:.0f} MB ONNX...")
        cmd = [
            TRTEXEC,
            f"--onnx={onnx_path}",
            "--fp16",
            "--memPoolSize=workspace:4096MiB",
            f"--saveEngine={engine_path}",
            "--monitorMemory",
        ]
        print(f"  $ {' '.join(os.path.basename(c) for c in cmd)}")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if r.returncode == 0 and os.path.exists(engine_path):
            esz = os.path.getsize(engine_path) / 1024**2
            print(f"  SUCCESS: {esz:.0f} MB engine")
            # Extract perf lines
            for line in (r.stdout + r.stderr).split("\n"):
                if any(k in line.lower() for k in ["throughput", "latency", "gpu compute", "enqueue"]):
                    print(f"  {line.strip()}")
        else:
            print(f"  FAILED. Last 10 lines:")
            output = (r.stdout + r.stderr).strip().split("\n")
            for line in output[-10:]:
                print(f"    {line.strip()}")

    print("\nDone.")


# ──────────────────────────────────────────────────────────────
# Step 4: Benchmark all available engines
# ──────────────────────────────────────────────────────────────
def do_bench():
    import torch
    import onnxruntime as ort
    banner("Step 4: Benchmark")
    gpu_info()

    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)

    pt_time = results.get("pytorch_fp16_ms", 80.0)
    print(f"PyTorch fp16 baseline: {pt_time:.1f}ms")

    # --- ONNX RT CUDA EP (fp16 ONNX) ---
    for onnx_path, label, inp_name, inp_shape, inp_dtype in [
        (ONNX_FP16, "full_wavlm_cuda_ep_fp16", "audio", (1, CHUNK_SAMPLES), np.float32),
        (ONNX_ENC_FP16, "encoder_cuda_ep_fp16", "features", (1, SEQ_LEN, HIDDEN), np.float32),
    ]:
        if not os.path.exists(onnx_path):
            continue
        print(f"\n--- {label} ---")
        try:
            sess = ort.InferenceSession(
                onnx_path,
                providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
            )
            inp = np.random.randn(*inp_shape).astype(inp_dtype)
            for _ in range(5):
                sess.run(None, {inp_name: inp})
            times = []
            for _ in range(20):
                t0 = time.perf_counter()
                sess.run(None, {inp_name: inp})
                times.append((time.perf_counter() - t0) * 1000)
            t = np.mean(times[5:])
            print(f"  {t:.1f}ms  ({pt_time/t:.2f}x vs PyTorch)")
            results[label] = round(t, 1)
            del sess
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")

    # --- ONNX RT TRT EP (fp16 ONNX → TRT at runtime) ---
    for onnx_path, label, inp_name, inp_shape in [
        (ONNX_FP16, "full_wavlm_trt_ep_fp16", "audio", (1, CHUNK_SAMPLES)),
        (ONNX_ENC_FP16, "encoder_trt_ep_fp16", "features", (1, SEQ_LEN, HIDDEN)),
    ]:
        if not os.path.exists(onnx_path):
            continue
        print(f"\n--- {label} ---")
        try:
            cache = os.path.join(TRT_CACHE, label)
            os.makedirs(cache, exist_ok=True)
            trt_opts = {
                "trt_fp16_enable": True,
                "trt_max_workspace_size": str(4 << 30),
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache,
            }
            sess = ort.InferenceSession(
                onnx_path,
                providers=[
                    ("TensorrtExecutionProvider", trt_opts),
                    ("CUDAExecutionProvider", {"device_id": 0}),
                    "CPUExecutionProvider",
                ],
            )
            print(f"  Providers: {sess.get_providers()}")
            inp = np.random.randn(*inp_shape).astype(np.float32)
            print("  Warmup (engine build may take a few minutes)...")
            for i in range(5):
                sess.run(None, {inp_name: inp})
                if i == 0:
                    print("  First inference done.")
            times = []
            for _ in range(20):
                t0 = time.perf_counter()
                sess.run(None, {inp_name: inp})
                times.append((time.perf_counter() - t0) * 1000)
            t = np.mean(times[5:])
            print(f"  {t:.1f}ms  ({pt_time/t:.2f}x vs PyTorch)")
            results[label] = round(t, 1)
            del sess
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")

    # Save all results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    banner("RESULTS SUMMARY")
    print(f"{'Method':<35} {'Time':>8} {'Speedup':>8}")
    print("-" * 55)
    print(f"{'PyTorch WavLM fp16 (baseline)':<35} {pt_time:>7.1f}ms {'1.00x':>8}")
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        if k == "pytorch_fp16_ms":
            continue
        sp = pt_time / v if v > 0 else 0
        print(f"{k:<35} {v:>7.1f}ms {sp:>7.2f}x")

    # Best
    bench_only = {k: v for k, v in results.items() if k != "pytorch_fp16_ms"}
    if bench_only:
        best = min(bench_only, key=bench_only.get)
        bt = bench_only[best]
        est = (250 - pt_time) + bt
        print(f"\nBEST: {best} = {bt:.1f}ms")
        print(f"Estimated full pipeline: ~{est:.0f}ms (current: 250ms)")
    print(f"\nResults saved to: {RESULTS_FILE}")


# ──────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────
def main():
    step = sys.argv[1] if len(sys.argv) > 1 else None

    if step == "export":
        do_export()
    elif step == "convert":
        do_convert()
    elif step == "build":
        do_build()
    elif step == "bench":
        do_bench()
    elif step is None:
        # Run all steps as separate subprocesses
        banner("WavLM TensorRT Engine Builder (orchestrator)")
        print("Each step runs in its own process for clean GPU memory.\n")

        steps = [
            ("export",  "Export WavLM to ONNX (fp32)"),
            ("convert", "Convert ONNX fp32 → fp16 (halves memory)"),
            ("build",   "Build TRT engines from fp16 ONNX"),
            ("bench",   "Benchmark all engines"),
        ]
        for name, desc in steps:
            print(f"\n{'─'*60}")
            print(f"  Step: {desc}")
            print(f"{'─'*60}")
            ok = run_step(name)
            if not ok:
                print(f"  ⚠ Step '{name}' had errors (continuing...)")

        # Print final results
        if os.path.exists(RESULTS_FILE):
            print(f"\n{'='*60}")
            print("  FINAL RESULTS")
            print(f"{'='*60}")
            with open(RESULTS_FILE) as f:
                results = json.load(f)
            for k, v in sorted(results.items(), key=lambda x: x[1]):
                print(f"  {k}: {v}ms")
    else:
        print(f"Unknown step: {step}")
        print("Usage: python build_trt_engine.py [export|convert|build|bench]")
        sys.exit(1)


if __name__ == "__main__":
    main()
