# MLHOMEWORK

Avazu CTR MLOps (SWE016)

End-to-end MLOps project for high-cardinality CTR prediction on the Avazu dataset. The pipeline uses hashed features with feature crosses, rebalancing, streaming training with no-leak validation split, MLflow tracking and model registry, and checkpointing.

## Setup
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Training
Baseline (sequential split, fixed rows):
```bash
.venv/bin/python src/train_baseline.py --train-rows 3000000 --val-rows 300000
```

Streaming (chunked, no-leak split):
```bash
.venv/bin/python src/train_streaming.py --chunk-size 100000 --max-train-chunks 30 --val-chunks 3 --checkpoint-every 5
```

## MLflow UI
Run locally:
```bash
.venv/bin/mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5002
```
Open: `http://127.0.0.1:5002`

## Outputs
- Models: `models/ctr_model_hashing.joblib`, `models/ctr_baseline_hashing.joblib`
- Metrics: `metrics/metrics.json`, `metrics/metrics_baseline.json`
- Checkpoints: `models/checkpoints/ckpt_chunk_*.joblib`

## Repository Structure
```
src/                training + feature code
scripts/            validate/register helpers
tests/              unit tests
training_summary.md latest results & evidence
```
