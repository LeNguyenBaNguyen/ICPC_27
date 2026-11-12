# Claim survival — anonymous research artifact

This release accompanies the current five-page manuscript in `manuscript/main.pdf`. The endpoint is first judge-detected contradiction, conditional on anchor support.

Start with `manuscript/main.pdf`, `EXPERIMENT_RESULTS.md`, `TWO_RATER_EVALUATION.md`, and `PPV_SENSITIVITY_VALIDATION.md`. The frozen historical records are in `artifact/results/`; later model replays, mutation checks and redraws are in `artifact/experiments/`; human labels and sensitivity analyses are in `artifact/validation/`. Analysis code is in `artifact/src/`.

Run `python artifact/src/check_artifact.py` after extraction to verify checksums and the frozen historical results without a network connection or model calls. The complete current manuscript and validation checks were also tested locally before packaging. This ZIP is for upload as a single GitHub file.
