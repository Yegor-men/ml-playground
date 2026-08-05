# ml-playground

Small, independent machine-learning experiments, largely revolving around text and image generation.

Each experiment lives in its own directory under `experiments/`. Its settings and hyperparameters are grouped near the
top of the main script, so experiments are run directly without command-line configuration:

```bash
python experiments/vae/main.py
```

Downloaded datasets are shared through the repository-level `data/` directory. Training scripts report progress with
`tqdm`, print epoch diagnostics, plot training and evaluation metrics, and finish with a qualitative visualization of
held-out examples.

The `local_regularization` directory contains several related comparisons that share `regularizer.py`; the other
experiments use `main.py` as their entry point.
