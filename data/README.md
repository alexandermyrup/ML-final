# Data

## Files in this folder

| File | Size | Source | Purpose |
|---|---|---|---|
| `consolidated_modeling_data.parquet` | 303 MB | **Fetch from GitHub release** (one-time) | Feature matrix + target for every model |
| `backtest_context.parquet` | 20 MB | Committed in this repo | Trade USD + corrected YES price for `Script 05` only — not a modeling feature |

## Get the modeling parquet

```bash
curl -L -o data/consolidated_modeling_data.parquet \
  https://github.com/alexandermyrup/ML-final/releases/download/data-v1/consolidated_modeling_data.parquet
echo "85f424975f8384590d8cd781d9353e2977a6cf336fa4bda01b1310f65e98cc87  data/consolidated_modeling_data.parquet" | shasum -a 256 -c
```

The notebook will refuse to run Script 01 if this file is missing.

## Quick schema reference

```python
import pandas as pd
df = pd.read_parquet("data/consolidated_modeling_data.parquet")
# (1,371,180 rows × 87 columns)

META_COLS = ["split", "market_id", "ts_dt", "timestamp"]
TARGET    = "bet_correct"   # binary, ~50.3% positive

train = df[df["split"] == "train"]   # 1,114,003 rows, 63 markets
test  = df[df["split"] == "test"]    #   257,177 rows, 10 markets

feature_cols = [c for c in df.columns if c not in META_COLS + [TARGET]]
# 82 candidate features; Script 01 reduces to 43 after dropping leakage-flagged cols
```

## Provenance

- Train/test split is **by market cohort**, not random. No market appears in both splits. This is the principal leakage defence — random splits would let the model peek at the same market's outcome.
- Train cohort ends at the 2026-02-28 strike event; test cohort ends 2026-03-31, well before the 2026-04-07 ceasefire upper bound.
- 12 wallet features (point-in-time, all prefixed `wallet_*` plus `days_from_first_usdc_to_t`) are already joined into the parquet. No separate wallet file needed.
- Build date: 2026-04-29.

## Leakage notes

`Script 01` enforces a `FORBIDDEN_LEAKY_COLS` blocklist:
- Causality / future-looking columns (lifetime stats that peek past trade time)
- Direction-encoding pairs that jointly determine `bet_correct` via an XOR formula whose mapping flips across market resolution types — these cause catastrophic test-set inversion on single-resolution cohorts
- Absolute-scale columns that leak market identity (deadline distance, market price level)

41 columns are dropped this way. `outputs/data/feature_cols.json` is written with the final 43 features used for modeling.

## Backtest sidecar

`backtest_context.parquet` supplies the raw trade fields kept out of the modelling matrix:
- `usd_amount` — true trade USD for liquidity-aware bet sizing
- `pre_yes_price_corrected` — YES-normalised pre-trade price for cost / edge math
- `taker`, `taker_direction`, `nonusdc_side` — for SELL-semantics diagnostics

`Script 01` validates row-for-row alignment between this sidecar and the modeling parquet. `Script 05` fails fast if the sidecar is missing.
