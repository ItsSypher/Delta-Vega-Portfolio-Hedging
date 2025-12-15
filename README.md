# Hedgesimple – Delta & Vega Hedging Simulator

Hedgesimple runs option hedging simulations comparing delta-only vs delta–vega strategies across multiple tickers, expiries, rebalancing cadences, and transaction-cost assumptions. It consumes pre-computed option surfaces with Greeks and produces tidy CSV/JSON outputs plus plotting helpers.

## Core Features
- Simulates ATM-straddle portfolios sized to initial capital, or custom portfolios via config.
- Compares `delta` and `delta_vega` hedges, each with/without transaction costs and multiple rebalance frequencies.
- Applies realistic transaction costs on traded notionals and tracks hedge P&L separately from portfolio moves.
- Parallelized execution with configurable worker count and subset/"quick" modes for fast iteration.
- Generates summary metrics (Sharpe, drawdown, P&L, cost impact) and per-day equity curves suitable for further analysis/visualisation.

## Repository Map
- Simulation entrypoint: `main.py`
- Configuration: `config/settings.yaml`
- Core logic: `src/data_loader.py`, `src/portfolio.py`, `src/hedger.py`, `src/simulator.py`
- Data (example): `data/raw/options_timeseries_filled.csv` (see `docs/DATA_FORMAT.md`)
- Outputs: `results/hedging_timeseries.csv`, `results/hedging_results.json`
- Plotting helpers: `processing/scripts/*.py` and `visualisation/seaborn_suite.py` (see `docs/visualisation.md`)

## Quick Start
```bash
# Run with default config
python main.py

# Fast smoke test: 1 ticker, 2 expiries, daily rebalance
python main.py --quick

# Limit scope
python main.py --tickers AAPL.O TSLA.O --max-expiries 3 --rebalance 1 5

# Delta-only, no transaction-cost runs
python main.py --delta-only --no-cost
```

The default config lives at `config/settings.yaml`. Use `--config path/to/file.yaml` to point at another file.

## Inputs
- Options surface with Greeks: CSV at `data/raw/options_timeseries_filled.csv` (columns described in `docs/DATA_FORMAT.md`).
- Configurable tickers, maturity dates, rebalance cadences, transaction cost pct, and portfolio construction.

## Outputs
- `results/hedging_timeseries.csv`: Daily snapshots with portfolio Greeks, hedge positions, and cumulative P&L per scenario.
- `results/hedging_results.json`: Aggregated stats (global, by ticker/expiry), cost impact, and optimal rebalance analysis.

## Scenario Naming
Scenarios follow `{hedge_type}_{cost_mode}_rebal_{frequency}d`, e.g., `delta_vega_with_cost_rebal_5d`. Each scenario tracks hedge trades, transaction costs, and equity path.

## Plotting
Use seaborn/matplotlib scripts to visualize outputs (see `docs/visualisation.md` for full CLI flags).

Examples:
```bash
# Cumulative P&L timeseries (default 5d cadence)
python processing/scripts/plot_pnl_timeseries.py \
  --csv results/hedging_timeseries.csv \
  --output processing/output/pnl_timeseries.png

# Per-ticker summary for a scenario
python processing/scripts/plot_ticker_summary.py \
  --csv results/hedging_timeseries.csv \
  --json results/hedging_results.json \
  --scenario delta_vega_with_cost_rebal_5d \
  --output processing/output/ticker_summary.png

# Optimal rebalance frequency visual
python processing/scripts/plot_rebalance_optimal.py \
  --json results/hedging_results.json \
  --metric total_pnl \
  --output processing/output/rebalance_optimal.png
```

## Configuration Highlights (`config/settings.yaml`)
- `tickers`, `maturity_dates`: universe of underlyings and expiries.
- `portfolio`: initial capital, portfolio type (`atm_straddle` auto-sized or `custom` positions), optional fixed quantity.
- `hedging`: transaction cost percent, hedge instrument preference, rebalance frequencies.
- `subset`: enable to cap tickers/expiries or restrict hedge types/cost modes/frequencies for faster runs; `--quick` sets a minimal subset via CLI.
- `parallel`: toggle threads/processes and worker count.

## Data Notes
- Data spans ~60 trading days before each expiry with $5 strike spacing around ATM.
- Mid prices are computed from bid/ask; Greeks/IV filled where missing (see `docs/DATA_FORMAT.md`).

## Requirements
Python 3 with pandas, numpy, PyYAML, seaborn, matplotlib (for plotting scripts). Install the stack with your preferred environment manager.
