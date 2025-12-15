"""Visualize metric vs rebalance frequency to highlight optimal cadence.

Defaults to with-cost scenarios, plotting total P&L vs rebalance_days.
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

from visualisation.seaborn_suite import load_rebalance_averages, plot_rebalance_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot optimal rebalancing frequency")
    parser.add_argument(
        "--json",
        default="results/hedging_results.json",
        help="Path to hedging_results.json",
    )
    parser.add_argument(
        "--metric",
        default="total_pnl",
        choices=["total_pnl", "total_pnl_pct", "sharpe_ratio", "max_drawdown_pct", "total_transaction_costs"],
        help="Metric to plot against rebalance frequency",
    )
    parser.add_argument(
        "--include-no-cost",
        action="store_true",
        help="Include no_cost scenarios in addition to with_cost",
    )
    parser.add_argument(
        "--output",
        default="processing/output/rebalance_optimal.png",
        help="Output image path (PNG)",
    )
    parser.add_argument(
        "--dump-csv",
        help="Optional path to save the tidy per-frequency data",
    )
    parser.add_argument(
        "--title",
        default="Rebalancing cadence vs metric",
        help="Plot title",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    df = load_rebalance_averages(json_path)
    if df.empty:
        raise ValueError("No per-frequency averages found in results JSON")

    ax = plot_rebalance_metric(
        df=df,
        metric=args.metric,
        title=args.title,
        with_cost_only=not args.include_no_cost,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Saved plot to {output_path}")

    if args.dump_csv:
        dump_path = Path(args.dump_csv)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dump_path, index=False)
        print(f"Saved plot data to {dump_path}")


if __name__ == "__main__":
    main()
