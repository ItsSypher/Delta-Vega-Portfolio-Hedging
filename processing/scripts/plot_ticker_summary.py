"""Plot per-ticker summary metrics for a chosen scenario.

Metrics shown: final cumulative P&L, max drawdown, realized volatility, average vega.
Default scenario: delta_vega_with_cost_rebal_5d.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualisation.seaborn_suite import (
    build_initial_map,
    load_timeseries,
    tidy_scenarios,
    ticker_metrics,
    plot_ticker_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot per-ticker hedging metrics")
    parser.add_argument(
        "--csv",
        default="results/hedging_timeseries.csv",
        help="Path to hedging time-series CSV",
    )
    parser.add_argument(
        "--json",
        default="results/hedging_results.json",
        help="Path to results JSON for initial values (optional)",
    )
    parser.add_argument(
        "--scenario",
        default="delta_vega_with_cost_rebal_5d",
        help="Scenario name to plot",
    )
    parser.add_argument(
        "--output",
        default="processing/output/ticker_summary.png",
        help="Output image path (PNG)",
    )
    parser.add_argument(
        "--metrics-csv",
        help="Optional CSV to save computed metrics",
    )
    parser.add_argument(
        "--sort-by",
        default="final_pnl",
        choices=["final_pnl", "max_drawdown_pct", "realized_vol", "avg_vega"],
        help="Metric to sort bars",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_timeseries(csv_path)
    long_df = tidy_scenarios(df, include_all=True)

    initial_map = None
    json_path = Path(args.json)
    if json_path.exists():
        initial_map = build_initial_map(json_path)

    metrics_df = ticker_metrics(
        long_df=long_df,
        base_df=df,
        scenario=args.scenario,
        initial_map=initial_map,
    )
    if metrics_df.empty:
        raise ValueError(f"No metrics computed for scenario {args.scenario}")

    fig = plot_ticker_summary(metrics_df, sort_by=args.sort_by)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")

    if args.metrics_csv:
        metrics_out = Path(args.metrics_csv)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_out, index=False)
        print(f"Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
