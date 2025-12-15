"""Plot cumulative P&L time-series from hedging_timeseries.csv.

Defaults to 5d scenarios to avoid clutter; --include-all overlays 1d/5d/10d.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualisation.seaborn_suite import load_timeseries, tidy_scenarios, plot_pnl_timeseries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot hedging P&L time-series")
    parser.add_argument(
        "--csv",
        default="results/hedging_timeseries.csv",
        help="Path to hedging time-series CSV",
    )
    parser.add_argument(
        "--output",
        default="processing/output/pnl_timeseries.png",
        help="Output image path (PNG)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include all rebalance frequencies (1d/5d/10d) instead of 5d-only",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Explicit scenario names to plot (overrides include-all filter)",
    )
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Average across tickers/expiries for smoother scenario curves",
    )
    parser.add_argument(
        "--roll",
        type=int,
        help="Optional rolling window (days) to smooth cumulative P&L",
    )
    parser.add_argument(
        "--start-date",
        help="Optional start date (YYYY-MM-DD) filter",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date (YYYY-MM-DD) filter",
    )
    parser.add_argument(
        "--title",
        default="Cumulative P&L",
        help="Plot title",
    )
    parser.add_argument(
        "--dump-csv",
        help="Optional path to save the tidy data used for plotting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = load_timeseries(csv_path)

    # Date filtering if provided
    if args.start_date:
        start = pd.to_datetime(args.start_date)
        df = df[df["date"] >= start]
    if args.end_date:
        end = pd.to_datetime(args.end_date)
        df = df[df["date"] <= end]

    long_df = tidy_scenarios(
        df,
        include_all=args.include_all,
        scenarios=args.scenarios,
        aggregate=args.aggregate,
        rolling=args.roll,
        rebase=True,
    )
    if long_df.empty:
        raise ValueError("No scenario data found after filtering")

    ax = plot_pnl_timeseries(long_df, title=args.title, use_smoothed=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Saved plot to {output_path}")

    if args.dump_csv:
        dump_path = Path(args.dump_csv)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(dump_path, index=False)
        print(f"Saved plot data to {dump_path}")


if __name__ == "__main__":
    main()
