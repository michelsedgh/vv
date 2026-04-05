# Voice Assistant Pipeline — Feasibility Analysis

## Executive Summary

Your proposed pipeline is **architecturally sound** but has **several critical issues** specific to your Jetson Orin Nano 8GB hardware, plus some logic/integration gaps. Below is a component-by-component breakdown.

---

## 1. PixIT / ToTaToNet (separation-ami-1.0) — SHOWSTOPPER ISSUE

### What it does
- Joint diarization + speech separation in one model
- Ingests 5s chunks of 16kHz mono audio
- Outputs: `diarization` tensor `(batch, 624 frames, 3 speakers)` + `sources` tensor `(batch, 80000 samples, 3 speakers)`
- Max 3 speakers per chunk (fine for home — <1% chance of >3 in a 5s window)

### The Problem: WavLM-Large
**ToTaToNet uses WavLM-Large as its feature extractor.** WavLM-Large is a 316M parameter transformer model.

Memory footprint estimate:
- WavLM-Large alone: ~1.2 GB in FP32, ~600 MB in FP16
- ToTaToNet separator (DPRNN) + diarization head: ~50-100 MB additional
- Plus the embedding model for clustering: ~100-300 MB (WeSpeaker/ECAPA-TDNN)
- Plus PyTorch/CUDA overhead: ~500 MB-1 GB
- **Total: 2-3+ GB GPU memory just for the PixIT pipeline**

Your Jetson Orin Nano has **8 GB shared** between CPU and GPU. After the OS, desktop environment, and other processes, you likely have 5-6 GB usable. PixIT alone would eat a massive chunk of that, leaving little for anything else (Whisper, LLM, DeepFilterNet, etc.).

### Additional Concerns
- `pyannote.audio[separation]` requires **asteroid** library as a dependency for the separation masking network
- The model was trained with pyannote.audio **3.3.0/3.3.2** — your current install is pyannote.audio 3.4.0, which *should* be backward-compatible but is untested with diart wrapping
- WavLM-Large inference latency on Jetson Orin Nano is **unknown** — could be 1-5+ seconds per 5s chunk, potentially too slow for real-time

### Verdict
**PixIT on Jetson Orin Nano is extremely risky.** The WavLM-Large backbone alone may make it infeasible. You should benchmark it before building anything on top.

---

## 2. Wrapping PixIT as a diart SegmentationModel — HARD BUT DOABLE

### How diart's SegmentationModel works
From the source code, diart expects:
```python
# SegmentationModel.__call__
# Input:  waveform: torch.Tensor, shape (batch, channels, samples)
# Output: torch.Tensor, shape (batch, frames, speakers)
```

PixIT's ToTaToNet returns a **tuple**: `(diarization, sources)`. Diart's `SegmentationModel` only expects the diarization tensor.

### What you'd need to do
1. Write a custom loader that loads `pyannote/separation-ami-1.0`
2. In the forward pass, call the model, return ONLY the diarization tensor to diart
3. Side-buffer the sources tensor keyed by chunk timestamp for later retrieval
4. This is the wrapper approach you described — it's correct in principle

### The tricky part
Diart's `SpeakerDiarization.__call__()` flow is:
```
segmentations = self.segmentation(batch)    # your wrapper returns diarization only
embeddings = self.embedding(batch, segmentations)  # standard embedding extraction
```

**Important subtlety**: Diart extracts speaker embeddings from the **original audio** weighted by the segmentation activations. It does NOT use the separated sources for embedding extraction. This means:
- The embedding model gets the original mixed audio, not clean separated sources
- PixIT's separated sources are only useful downstream (for Whisper), not for diart's clustering

This is actually fine for your pipeline — diart handles who-is-who via embeddings from original audio, and you use PixIT's separated sources for transcription. But understand that **diart's speaker embeddings come from the mixture, not from separated sources**.

### PixIT's own inference pipeline
The PixIT paper describes their OWN offline inference: they extract ECAPA-TDNN embeddings from **single-speaker regions of the original audio** (not separated sources), then do agglomerative clustering. This is conceptually similar to what diart does, except diart uses incremental online clustering instead of offline agglomerative clustering.

So **using PixIT inside diart** means:
- PixIT does segmentation + separation per chunk
- Diart does incremental clustering using embeddings from the mixture (weighted by PixIT's diarization output)
- You grab PixIT's separated sources on the side for downstream use

This should work. The diarization tensor from PixIT will be higher quality than from `pyannote/segmentation` alone (because PixIT was trained jointly), so diart's clustering should benefit.

---

## 3. Speaker Enrollment & Centroid Injection — FEASIBLE BUT REQUIRES CUSTOM CODE

### Current state of diart
- **PR #228** (DmitriyG228): Attempted implementation, but incomplete. Only implemented centroid *retrieval* (getting centroids out), NOT centroid *injection* (setting them before inference). juanmc2005 explicitly asked for this and it was never added. PR is stale since Jan 2024.
- **juanmc2005's suggestion** (Mar 2024): Add `freeze_centroids(centroids: list[int])` to `OnlineSpeakerClustering`, prevent updates in `identify()`. He said he'd merge a PR with this.
- **Neither feature exists in diart 0.9.2 today.**

### What you need to build
Looking at `OnlineSpeakerClustering` source code:

```python
# Key attributes:
self.centers: np.ndarray  # shape (max_speakers, embedding_dim), the centroid matrix
self.active_centers: set  # which center indices are in use
self.blocked_centers: set # which centers are blocked (unused in current code)
```

You need to add:
1. **`set_centroids(embeddings: dict[str, np.ndarray])`** — inject pre-enrolled speaker embeddings as initial centroids before inference starts
2. **`freeze_centroids(indices: list[int])`** — mark certain centroids as frozen so `update()` doesn't modify them
3. **Bounded drift** (your more sophisticated version) — instead of hard freeze, allow limited drift with rubber-band back to anchor

### Implementation approach
Modify `OnlineSpeakerClustering` (or subclass it):

```python
def set_centroids(self, embeddings: list[np.ndarray]):
    """Inject enrollment embeddings as initial centroids"""
    dim = embeddings[0].shape[0]
    self.init_centers(dim)
    for emb in embeddings:
        self.add_center(emb)
    # Store anchors for bounded drift
    self.anchors = {i: emb.copy() for i, emb in enumerate(embeddings)}
    self.max_drift = 0.3  # max cosine distance from anchor

def update(self, assignments, embeddings):
    """Modified update with bounded drift"""
    if self.centers is not None:
        for l_spk, g_spk in assignments:
            assert g_spk in self.active_centers
            if g_spk in self.anchors:
                # Bounded drift: update but rubber-band back
                new_center = self.centers[g_spk] + embeddings[l_spk]
                # Check drift from anchor
                drift = cosine_distance(new_center, self.anchors[g_spk])
                if drift > self.max_drift:
                    # Scale back toward anchor
                    alpha = self.max_drift / drift
                    new_center = self.anchors[g_spk] + alpha * (new_center - self.anchors[g_spk])
                self.centers[g_spk] = new_center
            else:
                # Stranger: update freely
                self.centers[g_spk] += embeddings[l_spk]
```

### Logic concern with your plan
Your plan says to use **WeSpeaker's `register/recognize` API** for enrollment. But WeSpeaker's Python API is a **standalone** tool — it has its own internal embedding store. You can't directly inject WeSpeaker's stored embeddings into diart's `OnlineSpeakerClustering` because:
1. WeSpeaker's `register()` stores embeddings internally in its own format
2. Diart's clustering uses whatever embedding model you configure (could be WeSpeaker-based or not)
3. **The enrollment embeddings MUST come from the same embedding model that diart uses for clustering**, otherwise cosine distances are meaningless

**Correct approach**: Use the **same** embedding model for both enrollment and live clustering. If you use `pyannote/wespeaker-voxceleb-resnet34-LM` in diart, then use that same model to extract enrollment embeddings (not WeSpeaker's CLI tool separately).

---

## 4. Embedding Model Choice — WeSpeaker vs TitaNet vs ECAPA-TDNN

### TitaNet (`nvidia/speakerverification_en_titanet_large`)
- Listed as supported in diart 0.9
- **REQUIRES NeMo** — and you've confirmed NeMo doesn't work on Jetson with PyTorch 2.3.0
- **ELIMINATED** for your hardware

### WeSpeaker (`pyannote/wespeaker-voxceleb-resnet34-LM` or `hbredin/wespeaker-voxceleb-resnet34-LM`)
- **Natively supported in diart 0.9** (listed in README)
- Does NOT require the `wespeaker` pip package — diart loads it via pyannote's `PretrainedSpeakerEmbedding` wrapper
- ResNet34-based, 256-dim embeddings
- Reasonable size (~25-50 MB), fast inference
- **BEST OPTION for your setup**

### SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)
- Already working in your current setup
- 192-dim embeddings
- Also supported in diart 0.9
- **GOOD FALLBACK**

### Recommendation
Use **WeSpeaker ResNet34** (`pyannote/wespeaker-voxceleb-resnet34-LM`) — it's the model diart was most recently optimized with, and it avoids the SpeechBrain 1.0 pinning issue.

For enrollment: load the same model via pyannote and extract embeddings from clean recordings of each family member. Don't use the `wespeaker` pip package separately.

---

## 5. DeepFilterNet — FEASIBILITY ON JETSON

### The issue
DeepFilterNet's core is written in **Rust** (libDF). The Python package (`deepfilternet` on PyPI) wraps the Rust binary. Pre-compiled wheels may not exist for **aarch64/ARM64**.

### Options
1. **PyPI wheel**: May or may not have an aarch64 build. You'd need to check.
2. **Build from source**: Requires Rust toolchain (`cargo`/`rustup`) installed on Jetson. Compiling Rust on ARM64 is slow but possible.
3. **Pure PyTorch reimplementation**: Someone made one (GitHub issue #430) but it was never merged. You could potentially use that fork.
4. **Alternative**: Use `noisereduce` (Python, simpler, but less effective) or `RNNoise` (has C bindings, lighter weight).

### Do you even need it?
PixIT was trained on AMI data with real room noise and reverberation. It's designed to handle noisy multi-speaker audio. Adding DeepFilterNet upstream might actually **hurt** PixIT's performance by removing acoustic cues the model expects.

**Recommendation**: Skip DeepFilterNet initially. Test PixIT's separation quality on raw mic input first. Only add noise reduction if separation quality is poor.

---

## 6. Source-to-Speaker Mapping — LOGIC ISSUE

Your plan says:
> After diart resolves global labels, look up the PixIT source buffer for that chunk. Use the diarization activation tensor to find which source index = which speaker.

### The problem
PixIT outputs are **locally permuted per chunk** — source index 0 in chunk N is NOT necessarily the same speaker as source index 0 in chunk N+1. The diarization tensor has the same local permutation. Diart resolves this by mapping local speakers to global speakers via clustering.

But diart's clustering operates on the **diarization tensor**, not the sources. After clustering, you know: "local speaker 2 in this chunk = global speaker 'Dad'". You then need to grab source index 2 from PixIT's output for that chunk.

### The mapping
The `SpeakerMap` returned by `OnlineSpeakerClustering.identify()` gives you exactly this mapping: `local_speaker_index → global_speaker_index`. You can use this to grab the right separated source.

**This part of your plan is correct**, but the implementation detail matters: you need access to the `SpeakerMap` object from inside the pipeline, which means you'd need to either:
- Subclass `SpeakerDiarization` to expose the map
- Or hook into the pipeline after clustering but before aggregation

---

## 7. Overall Pipeline Feasibility on Jetson Orin Nano 8GB

### Memory budget (rough estimates)

| Component | GPU Memory (FP32) | GPU Memory (FP16) |
|-----------|-------------------|-------------------|
| PixIT (WavLM-Large + ToTaToNet) | ~1.5 GB | ~800 MB |
| WeSpeaker ResNet34 embedding | ~50 MB | ~25 MB |
| Faster-Whisper (small/base) | ~500 MB | ~250 MB |
| DeepFilterNet | ~100 MB | ~50 MB |
| PyTorch/CUDA overhead | ~800 MB | ~800 MB |
| **Total** | **~3.0 GB** | **~1.9 GB** |

With 8 GB shared and OS taking ~2 GB, you have ~6 GB. In FP16, the core pipeline might fit (~2 GB), but:
- WavLM-Large inference latency is the real unknown
- You still need memory for the LLM response (even via API, the audio buffers eat RAM)

### Latency budget

| Component | Estimated latency per step |
|-----------|--------------------------|
| PixIT (5s chunk, WavLM-Large, Jetson) | 2-8s (UNKNOWN, likely too slow) |
| WeSpeaker embedding extraction | ~50-100ms |
| Diart clustering | ~10ms |
| Faster-Whisper (on utterance end) | 1-3s for typical utterance |

The **critical unknown** is PixIT inference time on Jetson. WavLM-Large is a 24-layer transformer — on a Jetson Orin Nano, this could easily be 3-8 seconds per 5s chunk, which would make real-time streaming impossible (you need <500ms per step).

---

## 8. Critical Logic Flaws & Corrections

### Flaw 1: WeSpeaker register/recognize is NOT how enrollment should work
- WeSpeaker's Python API is standalone — its internal embedding store doesn't connect to diart
- **Fix**: Use diart's embedding model directly for both enrollment and live processing

### Flaw 2: TitaNet is NOT usable on your hardware
- Requires NeMo which needs PyTorch 2.4+ (nn.Buffer) or has huggingface_hub incompatibilities
- **Fix**: Use WeSpeaker ResNet34 (natively supported in diart, no NeMo dependency)

### Flaw 3: PixIT + WavLM-Large may be too heavy for Jetson
- 316M param feature extractor + DPRNN separator
- **Fix**: Benchmark first. Fallback: use standard `pyannote/segmentation-3.0` for diarization only, and run a lighter separation model separately (or skip separation, extract from diarization regions of original audio)

### Flaw 4: DeepFilterNet may be counterproductive
- PixIT was trained on noisy real-world data
- Denoising upstream may remove information the model needs
- Rust compilation on ARM64 is non-trivial
- **Fix**: Skip it initially, add only if needed

### Flaw 5: diart version pinning
- diart 0.9.2 requires `pyannote.audio>=2.1.1` (no upper bound)
- PixIT requires `pyannote.audio[separation]==3.3.2`
- Your current install is pyannote.audio 3.4.0
- The `[separation]` extra pulls in `asteroid` library
- **Fix**: You may need to pin pyannote.audio to 3.3.2 for PixIT compatibility, and verify diart still works

---

## 9. Recommended Approach (Phased)

### Phase 1: Validate PixIT on Jetson (DO THIS FIRST)
```bash
pip install pyannote.audio[separation]==3.3.2
```
Then run a simple benchmark:
```python
from pyannote.audio import Model
import torch, time

model = Model.from_pretrained("pyannote/separation-ami-1.0")
model.to(torch.device("cuda"))

waveform = torch.randn(1, 1, 80000).to("cuda")  # 5s at 16kHz
with torch.inference_mode():
    start = time.time()
    diarization, sources = model(waveform)
    elapsed = time.time() - start
    print(f"Inference: {elapsed:.2f}s")
    print(f"Diarization shape: {diarization.shape}")
    print(f"Sources shape: {sources.shape}")
```

**If this takes >2 seconds or OOMs, PixIT is not viable on your Jetson.**

### Phase 2: If PixIT works — diart integration
1. Install diart 0.9.2 with the WeSpeaker embedding model
2. Write the PixIT SegmentationModel wrapper (return diarization only, side-buffer sources)
3. Write the `set_centroids()` + bounded drift modification to `OnlineSpeakerClustering`
4. Write the source-to-speaker mapping layer
5. Test with mic input

### Phase 2 (Alternative): If PixIT is too heavy
Fall back to your existing setup:
1. Standard `pyannote/segmentation-3.0` in diart for diarization
2. WeSpeaker ResNet34 for embeddings
3. Custom centroid injection for enrollment
4. **No speech separation** — instead, extract speaker audio from original using diarization time segments
5. Feed those segments to Whisper (noisy but functional)

This fallback loses the clean separated sources but keeps everything else and will definitely run on Jetson.

### Phase 3: Enrollment system
- Record each family member (20-60s, clean, same mic/room)
- Extract embeddings using the same WeSpeaker model diart uses
- Inject as initial centroids with bounded drift
- Test recognition accuracy

---

## 10. Summary of What Exists vs What You'd Build

| Component | Status |
|-----------|--------|
| diart streaming framework | pip install, works |
| PixIT model (separation-ami-1.0) | pip install, **untested on Jetson** |
| PixIT → diart wrapper | **YOU BUILD** (nobody has done this) |
| WeSpeaker embedding in diart | Works out of box |
| Centroid injection (set_centroids) | **YOU BUILD** (PR #228 is incomplete) |
| Bounded drift | **YOU BUILD** (doesn't exist anywhere) |
| Source-to-speaker mapping | **YOU BUILD** |
| DeepFilterNet on Jetson | **Risky** (Rust+ARM64), skip initially |
| Per-speaker EOT detection | **YOU BUILD** |
| TitaNet in diart | **BLOCKED** (NeMo incompatible with your PyTorch) |
