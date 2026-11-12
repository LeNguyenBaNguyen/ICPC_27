# Reproducing the archived checks

From the extracted `claim-survival-artifact` directory, run:

```text
python artifact/src/check_artifact.py
```

This standard-library check verifies every packaged checksum, the frozen main estimate, 400 submitted A1 labels and 1,800 supplemental cached responses. It uses no network, GPU, credentials or fresh model calls. For analysis details and caveats, read `EXPERIMENT_RESULTS.md`, `TWO_RATER_EVALUATION.md`, `PPV_SENSITIVITY_VALIDATION.md`, and `artifact/MEASUREMENT_PROTOCOL.md`. The manuscript source and generated inputs are in `manuscript/`.
