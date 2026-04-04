#!/usr/bin/env bash
# Re-download MeanFlow-TSE noisy checkpoints (same files as MeanFlow README / Google Drive).
set -euo pipefail
cd "$(dirname "$0")"
python3 -m gdown --fuzzy "https://drive.google.com/file/d/1_ngNHWx2ClfUciyBhOmC8BD2VCuj0hN3/view?usp=share_link" -O best-noisy-weights.ckpt
python3 -m gdown --fuzzy "https://drive.google.com/file/d/1JJkI85nrKyNG0ZPjfat-KgUz-dbG2WKP/view?usp=share_link" -O t-predictor-noisy-weights.ckpt
echo "Done. Ensure config.yaml uses config_MeanFlowTSE_noisy.yaml and best-noisy-weights.ckpt."
