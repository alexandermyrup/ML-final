# ML-final — Polymarket Mispricing Detection

CBS Machine Learning & Deep Learning final project. Predicts whether individual Polymarket trades will resolve in the trader's favour, using 8 supervised models with GroupKFold cross-validation, isotonic calibration, bootstrap confidence intervals, permutation importance, SHAP on top picks, and a capital-aware backtest with a naive consensus baseline as falsification.

The full pipeline is bundled into a single Jupyter notebook (`v2scripts.ipynb`) that runs top-to-bottom.

---

## Quick start

Easiest path — just open `v2scripts.ipynb` in Jupyter and Run All. The notebook auto-downloads the 303 MB modeling parquet + 20 MB backtest sidecar from this repo's [`data-v1` release](https://github.com/alexandermyrup/ML-final/releases/tag/data-v1) on first run, with SHA-256 verification. No manual `curl` step needed.

```bash
# Optional: install deps if not already in your env
pip install -r requirements.txt

# Launch — that's it
jupyter notebook v2scripts.ipynb
```

End-to-end runtime is **~80 minutes** on a modern laptop, dominated by Script 03 (training, ~60 min) and Script 02 (Isolation Forest fit, ~5 min).

If you prefer to download the data manually (e.g. before opening Jupyter):

```bash
curl -L -o data/consolidated_modeling_data.parquet \
  https://github.com/alexandermyrup/ML-final/releases/download/data-v1/consolidated_modeling_data.parquet
curl -L -o data/backtest_context.parquet \
  https://github.com/alexandermyrup/ML-final/releases/download/data-v1/backtest_context.parquet
```

---

## What's in this repo

```
ML-final/
├── README.md                       <- this file
├── requirements.txt                <- pinned Python dependencies
├── v2scripts.ipynb                 <- the entire pipeline (8 stages in one notebook)
└── data/
    ├── README.md                   <- data format + provenance
    └── backtest_context.parquet    <- 20 MB sidecar, included directly
```

The 303 MB `consolidated_modeling_data.parquet` is **not** committed to git (too large). It lives on the [GitHub Release `data-v1`](https://github.com/alexandermyrup/ML-final/releases/tag/data-v1) and the one-line `curl` above fetches it. SHA-256 is provided for verification.

After running the notebook, generated artifacts land in `outputs/` (gitignored):
- `outputs/data/` — leakage report, feature list, scaler, IsoForest scores
- `outputs/models/<name>/` — OOF + raw + calibrated test predictions, per-model metrics
- `outputs/metrics/` — comparison table, calibration summary, AUC CIs, permutation importance, reliability diagrams
- `outputs/backtest/` — sensitivity grid, falsification verdict, diagnostics, headline ROI heatmap
- `outputs/tuning/<name>/` — Optuna study history, best params, tuned predictions (only if you flip `RUN_TUNING = True`)

---

## The 8 stages

| Stage | Purpose | Runtime |
|---|---|---|
| Setup | Imports, paths, plotting theme | <1 sec |
| 01 Data prep | Load parquet, split by market cohort, run 8 leakage / sanity checks, write canonical feature list | ~30 sec |
| 02 Features | Feature taxonomy, train-only `StandardScaler`, IsolationForest anomaly score | ~5 min |
| 03 Train models | 7-8 models (LogReg L1/L2, Decision Tree, RF, HistGBM, LightGBM, PCA→LogReg, sklearn MLP) with GroupKFold CV + complexity benchmark | ~60 min |
| 04 Calibration | Isotonic on OOF + reliability + bootstrap CIs + paired-bootstrap pairwise tests + permutation importance + SHAP on top picks | ~10 min |
| 05 Backtest | Capital-aware sim across 10 strategies × 3×3×2 sensitivity grid + naive baseline + headline ROI heatmap + residual-edge / consensus / SELL diagnostics | ~5 min |
| 06 Tuning (gated) | Optuna TPE for RF, HistGBM, MLP, LightGBM with GroupKFold or holdout + MedianPruner | ~30-60 min per model |
| Appendix (gated) | `promote_tuned_preds` bridge: copy tuned preds into the canonical slots for re-running 04 + 05 | <5 sec |

Stages 06 and the Appendix are gated behind `RUN_TUNING = False` / `RUN_PROMOTE = False` flags so they no-op by default. Flip to `True` if you want to run them.

---

## Dataset

- **1,371,180** Polymarket trades, 87 columns
- **1,114,003** train + **257,177** test, split by market cohort (no market spans both splits — leakage defence)
- 63 train markets, 10 test markets, zero overlap
- Target: `bet_correct` (binary, ~50.3% positive in both splits)
- Train cohort ends at the 2026-02-28 strike event; test cohort runs through 2026-03-31
- Build date: 2026-04-29

See `data/README.md` for full schema + feature taxonomy.

---

## Reproducibility

- `RANDOM_SEED = 42` (every model factory and `np.random` call)
- `N_FOLDS = 5` for `GroupKFold(group=market_id)` CV
- Every script writes to `outputs/...` with idempotent file paths
- All `predict_proba` outputs are saved (`outputs/models/<m>/preds_*.{npy,npz}`) so 04 and 05 can re-run without retraining

---

## Authors

CBS MSc Business Administration & Data Science, Spring 2026.

Course: KAN-CDSCO2004U Machine Learning and Deep Learning.
