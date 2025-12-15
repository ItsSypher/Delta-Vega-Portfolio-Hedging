# Hedging Visualisation (seaborn)

New plotting utilities live in visualisation/seaborn_suite.py with two CLI wrappers in processing/scripts.

## Requirements
- seaborn, matplotlib, pandas, numpy

## Quick commands

### P&L timeseries (default: 5d rebalance only)
```bash
python processing/scripts/plot_pnl_timeseries.py \
  --csv results/hedging_timeseries.csv \
  --output processing/output/pnl_timeseries.png
```

Include all rebalance frequencies (1d/5d/10d):
```bash
python processing/scripts/plot_pnl_timeseries.py \
  --include-all \
  --title "Cumulative P&L (all cadences)"
```

Key args:
- `--csv` path to hedging_timeseries.csv
- `--output` PNG path (300 DPI)
- `--include-all` include 1d/10d scenarios (default 5d-only)
- `--scenarios` explicit scenario names to plot
- `--aggregate` average across tickers/expiries per scenario/day
- `--roll N` rolling window (days) to smooth rebased curves
- `--start-date/--end-date` optional date filters (YYYY-MM-DD)
- `--dump-csv` save tidy data (includes `cumulative_pnl_rebased` and `cumulative_pnl_smooth`)
- Plots are rebased to start at zero per scenario; legend on the right

### Per-ticker summary (default scenario delta_vega_with_cost_rebal_5d)
```bash
python processing/scripts/plot_ticker_summary.py \
  --csv results/hedging_timeseries.csv \
  --json results/hedging_results.json \
  --output processing/output/ticker_summary.png
```

Save metrics as CSV too:
```bash
python processing/scripts/plot_ticker_summary.py \
  --metrics-csv processing/output/ticker_summary_metrics.csv
```

Key args:
- `--csv` hedging_timeseries.csv
- `--json` hedging_results.json (for initial capital map)
- `--scenario` scenario to summarize (default delta_vega_with_cost_rebal_5d)
- `--output` PNG path (300 DPI)
- `--metrics-csv` save computed metrics
- `--sort-by` one of final_pnl | max_drawdown_pct | realized_vol | avg_vega

## What gets computed
- Timeseries: cumulative P&L from day 0, grouped by scenario; default 5d-only keeps plots uncluttered.
- Per ticker: final cumulative P&L, max drawdown (based on equity = initial + cumulative P&L), realized underlying volatility (annualized log-return std), average portfolio vega exposure.

## Scenario naming
Format: `{hedge_type}_{cost_mode}_rebal_{frequency}d`, e.g. `delta_vega_with_cost_rebal_5d`. Use the `--scenarios` flag in plot_pnl_timeseries.py to plot a custom subset.

## Plotting considerations applied
- Legends are placed outside the plot (right side) to keep lines/bars unobstructed; layout tightened with bbox_inches="tight".
- Output resolution is 300 DPI for all scripts (PNG saves).
- P&L timeseries defaults:
  - Rebased per-scenario to start at zero (removes day-0 cost offsets in with_cost lines).
  - Optional aggregation across tickers/expiries for smoother scenario curves (`--aggregate`).
  - Optional rolling mean smoothing (`--roll N`, e.g., 5-day) on rebased curves for readability.
  - Dump CSV includes `cumulative_pnl_rebased` and `cumulative_pnl_smooth` for traceability.
- Rebalance optimal plot:
  - Default filters to with_cost scenarios; use `--include-no-cost` to overlay all.
  - Hue encodes both hedge_type and cost_mode (e.g., "delta | with_cost") so with_cost lines are visible.
  - Annotations mark the best cadence per series for the chosen metric.
- Per-ticker summary:
  - Uses 300 DPI output; metrics CSV dump available via `--metrics-csv`.

### Optimal rebalancing frequency
```bash
python processing/scripts/plot_rebalance_optimal.py \
  --json results/hedging_results.json \
  --metric total_pnl \
  --output processing/output/rebalance_optimal.png
```

Flags:
- `--metric` one of total_pnl, total_pnl_pct, sharpe_ratio, max_drawdown_pct, total_transaction_costs
- `--include-no-cost` to overlay no-cost alongside with-cost (default hides no-cost)
- `--dump-csv` save the tidy per-frequency data

Plot shows metric vs rebalance frequency with annotations of the best cadence per hedge type.
