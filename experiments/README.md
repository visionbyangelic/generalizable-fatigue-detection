# Experiments — Results-Logging Convention

One folder per run: `experiments/runs/YYYY-MM-DD_<model>_<protocol>_<labels>_s<seed>/`

Each run folder must contain:
- `config.yaml` — exact config used (copied from `configs/`, frozen)
- `metrics.csv` — per-participant recall / precision / F1 / balanced accuracy + aggregate
- `notes.md` — split definition, leakage checklist (normalisation train-only? tuning train-only? no overlapping windows across split? held-out participant untouched?), seed, environment (local/Kaggle/Colab), git commit hash

Naming: model ∈ {cnn, baseline, multimodal}, protocol ∈ {within, session-out, loso}, labels ∈ {subjective, performance, session}

No result counts for the project unless it has a run folder following this convention.
