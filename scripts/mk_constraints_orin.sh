#!/usr/bin/env bash
# Write constraints_orin.txt from installed torch / torchvision / torchaudio.
# Use: pip install -r requirements_orin_meanflow.txt -c constraints_orin.txt
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=constraints_orin.txt
python3 << 'PY' > "$OUT"
import importlib

for pkg in ("torch", "torchvision", "torchaudio"):
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}=={m.__version__}")
    except Exception:
        pass
PY
if [[ ! -s "$OUT" ]]; then
  echo "No torch/torchvision/torchaudio found. Install NVIDIA PyTorch first, then re-run."
  rm -f "$OUT"
  exit 1
fi
echo "Wrote $OUT:"
cat "$OUT"
echo ""
echo "Next: pip install -r requirements_orin_meanflow.txt -c constraints_orin.txt"
