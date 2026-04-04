# MeanFlow-TSE weights

1. Open the [MeanFlow-TSE README](https://github.com/rikishimizu/MeanFlow-TSE) → **Pretrained Models** → Google Drive folder.

2. **Noisy checkpoints (default in this repo’s `config.yaml`):** from Drive, or re-fetch with:

   ```bash
   pip install gdown   # if needed
   bash models/meanflow/download_weights.sh
   ```

   That writes `best-noisy-weights.ckpt` and `t-predictor-noisy-weights.ckpt` here. Use `meanflow_tse/config/config_MeanFlowTSE_noisy.yaml` + `checkpoint: models/meanflow/best-noisy-weights.ckpt`.

3. **Clean checkpoints:** download **`best.ckpt`** (clean) into `models/meanflow/best.ckpt` and set `config.yaml` to `config_MeanFlowTSE_clean.yaml` and that path.

4. **t-predictor:** `config.yaml` sets `meanflow.use_t_predictor: true` and `t_predictor_checkpoint` → same **ECAPAMLP** path as `eval_steps.py` / `inference_sample.ipynb` (Lightning ckpt, `model.` prefix stripped on load). Set `use_t_predictor: false` to fall back to **alpha = 0.5** (HALF).

If the main checkpoint is missing, the server will refuse to load the model and print the path it expects.
