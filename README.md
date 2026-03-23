# Experiment Code

This folder contains a self-contained implementation of the formulas used in the paper:

- `algorithms.py`: `FCM`, `PFCM`, `KPFCM`, `C-PFCM`, `C-KPFCM`
- `synthetic_data.py`: reproducible complex-background segmentation cases
- `evaluation.py`: segmentation accuracy and label remapping
- `plotting.py`: result figures
- `run_experiments.py`: full experiment pipeline

Run:

```bash
python3 code/run_experiments.py
```

Outputs are written to:

- `results/data`
- `results/figures`
- `results/report`
