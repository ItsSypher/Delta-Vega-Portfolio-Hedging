"""
Reusable seaborn-based helpers for hedging visualization.

Functions cover:
- Loading and tidying hedging time-series output
- Computing realized volatility, drawdowns, and vega exposure
- Plotting P&L time-series and per-ticker summaries

Defaults show 5d rebalance scenarios; callers can include all.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Scenario name pattern: {hedge_type}_{cost_mode}_rebal_{freq}d
SCENARIO_RE = re.compile(r"^(delta|delta_vega)_(no_cost|with_cost)_rebal_(\d+)d$")

BASE_COLUMNS = [
    "ticker",
    "expiry",
    "date",
    "dte",
    "underlying_price",
    "portfolio_value",
    "portfolio_delta",
    "portfolio_vega",
    "portfolio_gamma",
    "portfolio_theta",
]

SCENARIO_METRICS = [
    "shares",
    "options",
    "daily_pnl",
    "hedge_pnl",
    "transaction_cost",
    "cumulative_pnl",
    "cumulative_costs",
]


@dataclass(frozen=True)
class ScenarioParts:
    hedge_type: str
    cost_mode: str
    rebalance_days: int

    @property
    def label(self) -> str:
        cost_label = "with cost" if self.cost_mode == "with_cost" else "no cost"
        return f"{self.hedge_type.replace('_', '-')} | {cost_label} | {self.rebalance_days}d"


def _parse_scenario_name(name: str) -> Optional[ScenarioParts]:
    match = SCENARIO_RE.match(name)
    if not match:
        return None
    hedge_type, cost_mode, freq = match.groups()
    return ScenarioParts(hedge_type=hedge_type, cost_mode=cost_mode, rebalance_days=int(freq))


def load_timeseries(csv_path: Path | str) -> pd.DataFrame:
    """Load hedging time-series CSV with parsed dates."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def discover_scenarios(df: pd.DataFrame) -> List[str]:
    """Find scenario prefixes present in the wide-form CSV."""
    prefixes = set()
    for col in df.columns:
        for metric in SCENARIO_METRICS:
            suffix = f"_{metric}"
            if col.endswith(suffix):
                prefix = col[: -len(suffix)]
                if SCENARIO_RE.match(prefix):
                    prefixes.add(prefix)
    return sorted(prefixes)


def tidy_scenarios(
    df: pd.DataFrame,
    include_all: bool = False,
    scenarios: Optional[Iterable[str]] = None,
    aggregate: bool = False,
    rolling: Optional[int] = None,
    rebase: bool = True,
) -> pd.DataFrame:
    """
    Convert wide scenario columns into tidy long form with scenario parts.

    Args:
        df: DataFrame from load_timeseries
        include_all: If False, keeps only 5d rebalance scenarios
        scenarios: Optional explicit scenario list to keep
    """
    scenario_names = scenarios or discover_scenarios(df)
    if not include_all:
        scenario_names = [s for s in scenario_names if s.endswith("rebal_5d")]

    long_frames: List[pd.DataFrame] = []
    for scenario in scenario_names:
        parts = _parse_scenario_name(scenario)
        if parts is None:
            continue

        col_map = {
            metric: f"{scenario}_{metric}" for metric in SCENARIO_METRICS if f"{scenario}_{metric}" in df.columns
        }
        if not col_map:
            continue

        sub = df[BASE_COLUMNS + list(col_map.values())].rename(columns={v: k for k, v in col_map.items()})
        sub = sub.assign(
            scenario=scenario,
            hedge_type=parts.hedge_type,
            cost_mode=parts.cost_mode,
            rebalance_days=parts.rebalance_days,
        )
        long_frames.append(sub)

    if not long_frames:
        return pd.DataFrame()

    long_df = pd.concat(long_frames, ignore_index=True)
    long_df.sort_values(["ticker", "expiry", "date", "scenario"], inplace=True)
    long_df["days_since_start"] = (
        long_df.groupby(["ticker", "expiry"])["date"].transform(lambda s: (s - s.min()).dt.days)
    )
    # Rebase per scenario so curves start at zero
    if rebase:
        long_df["cumulative_pnl_rebased"] = long_df.groupby("scenario")["cumulative_pnl"].transform(
            lambda s: s - s.iloc[0]
        )
    else:
        long_df["cumulative_pnl_rebased"] = long_df["cumulative_pnl"]

    if aggregate:
        # Average across ticker/expiry for smoother curves per scenario/day
        long_df = (
            long_df.groupby(["scenario", "hedge_type", "cost_mode", "rebalance_days", "days_since_start"], as_index=False)[
                ["cumulative_pnl_rebased", "hedge_pnl", "transaction_cost"]
            ]
            .mean()
        )

    # Optional rolling mean for smoothing
    if rolling and rolling > 1:
        long_df["cumulative_pnl_smooth"] = (
            long_df.groupby("scenario")["cumulative_pnl_rebased"].transform(lambda s: s.rolling(rolling, min_periods=1).mean())
        )
    else:
        long_df["cumulative_pnl_smooth"] = long_df["cumulative_pnl_rebased"]
    long_df["scenario_label"] = long_df.apply(
        lambda r: _parse_scenario_name(r["scenario"]).label if _parse_scenario_name(r["scenario"]) else r["scenario"],
        axis=1,
    )
    return long_df


def compute_realized_vol(timeseries: pd.DataFrame) -> pd.DataFrame:
    """Realized annualized vol per ticker from underlying_price log returns."""
    ts = timeseries.sort_values(["ticker", "expiry", "date"]).copy()
    ts["log_ret"] = ts.groupby(["ticker", "expiry"])["underlying_price"].transform(lambda s: np.log(s).diff())
    agg = ts.groupby("ticker")["log_ret"].std() * np.sqrt(252)
    return agg.reset_index().rename(columns={"log_ret": "realized_vol"})


def _max_drawdown_from_curve(curve: pd.Series, initial_value: float) -> float:
    equity = initial_value + curve.fillna(0)
    running_max = equity.cummax()
    drawdown = (running_max - equity) / running_max.replace(0, np.nan)
    return float(drawdown.max()) if len(drawdown) else 0.0


def build_initial_map(results_json: Path | str) -> Dict[Tuple[str, str], float]:
    """Map (ticker, expiry) to initial_portfolio_value from results JSON."""
    with open(results_json, "r") as f:
        data = json.load(f)
    mapping: Dict[Tuple[str, str], float] = {}
    for ticker, expiry_results in data.get("results_by_ticker", {}).items():
        for expiry, payload in expiry_results.items():
            initial_val = payload.get("initial_portfolio_value")
            if initial_val is not None:
                mapping[(ticker, expiry)] = float(initial_val)
    return mapping


def ticker_metrics(
    long_df: pd.DataFrame,
    base_df: pd.DataFrame,
    scenario: str,
    initial_map: Optional[Dict[Tuple[str, str], float]] = None,
) -> pd.DataFrame:
    """
    Compute per-ticker metrics for a single scenario.

    Metrics:
    - final cumulative P&L
    - max drawdown (pct of peak equity)
    - realized volatility (from underlying prices)
    - average vega exposure
    """
    if long_df.empty:
        return pd.DataFrame()

    scenario_df = long_df[long_df["scenario"] == scenario].copy()
    if scenario_df.empty:
        return pd.DataFrame()

    # Realized vol per ticker
    realized = compute_realized_vol(base_df)

    records = []
    for ticker, group in scenario_df.groupby("ticker"):
        final_pnl = (
            group.sort_values("date")["cumulative_pnl"].dropna().iloc[-1]
            if group["cumulative_pnl"].notna().any()
            else 0.0
        )

        # Determine initial value
        initial_guess = None
        if initial_map:
            expiries = group["expiry"].unique()
            if len(expiries) == 1:
                key = (ticker, expiries[0])
                initial_guess = initial_map.get(key)
        if initial_guess is None:
            initial_guess = (
                base_df[(base_df["ticker"] == ticker) & (base_df["expiry"] == group["expiry"].iloc[0])][
                    "portfolio_value"
                ]
                .iloc[0]
            ) if not base_df.empty else 0.0

        max_dd = _max_drawdown_from_curve(group.sort_values("date")["cumulative_pnl"], initial_guess)

        avg_vega = (
            base_df[base_df["ticker"] == ticker]["portfolio_vega"].mean()
            if "portfolio_vega" in base_df.columns
            else np.nan
        )

        realized_vol = (
            realized[realized["ticker"] == ticker]["realized_vol"].iloc[0]
            if not realized.empty and ticker in realized["ticker"].values
            else np.nan
        )

        records.append(
            {
                "ticker": ticker,
                "scenario": scenario,
                "final_pnl": float(final_pnl),
                "max_drawdown_pct": float(max_dd * 100),
                "realized_vol": float(realized_vol) if pd.notna(realized_vol) else np.nan,
                "avg_vega": float(avg_vega) if pd.notna(avg_vega) else np.nan,
            }
        )

    return pd.DataFrame(records)


def plot_pnl_timeseries(
    long_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "P&L timeseries",
    y_label: str = "Cumulative P&L",
    use_smoothed: bool = True,
) -> plt.Axes:
    """Line plot of cumulative P&L vs days since start for selected scenarios."""
    if long_df.empty:
        raise ValueError("No data to plot")

    sns.set_theme(style="whitegrid")
    ax = ax or plt.gca()
    ax.figure.set_size_inches(10, 6)

    palette = {
        "delta": "#1f77b4",
        "delta_vega": "#d62728",
    }
    style_map = {"no_cost": "solid", "with_cost": "dashed"}

    for scenario, sdf in long_df.groupby("scenario"):
        parts = _parse_scenario_name(scenario)
        if parts is None:
            continue
        hue = palette.get(parts.hedge_type, "#4c4c4c")
        linestyle = style_map.get(parts.cost_mode, "solid")
        label = parts.label
        sdf_sorted = sdf.sort_values("days_since_start")
        ycol = "cumulative_pnl_smooth" if use_smoothed and "cumulative_pnl_smooth" in sdf_sorted else "cumulative_pnl_rebased"
        ax.plot(
            sdf_sorted["days_since_start"],
            sdf_sorted[ycol],
            label=label,
            color=hue,
            linestyle=linestyle,
            linewidth=1.8,
        )

    ax.set_xlabel("Days since hedge start")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title="Scenario", frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    ax.figure.tight_layout()
    return ax


def plot_ticker_summary(
    metrics_df: pd.DataFrame,
    sort_by: str = "final_pnl",
    metrics: Sequence[Tuple[str, str]] = (
        ("final_pnl", "Final P&L ($)"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("realized_vol", "Realized Vol (ann.)"),
        ("avg_vega", "Avg Vega Exposure"),
    ),
    height: float = 2.2,
    width: float = 6.5,
) -> plt.Figure:
    """Create horizontal bar panels for per-ticker metrics."""
    if metrics_df.empty:
        raise ValueError("No metrics to plot")

    ordered = metrics_df.sort_values(sort_by, ascending=False)
    tickers = ordered["ticker"].unique().tolist()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(len(metrics), 1, figsize=(width, height * len(metrics)), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        sns.barplot(data=ordered, y="ticker", x=col, order=tickers, ax=ax, color="#4c78a8")
        ax.set_title(label)
        ax.axvline(0, color="#888", linewidth=1, linestyle="--")
        ax.set_ylabel("")
    axes[-1].set_xlabel("Value")
    fig.tight_layout()
    return fig


def load_rebalance_averages(results_json: Path | str) -> pd.DataFrame:
    """
    Load per-frequency averages from hedging_results.json into tidy form.

    Returns columns: scenario, hedge_type, cost_mode, rebalance_days,
    total_pnl, total_pnl_pct, sharpe_ratio, max_drawdown_pct, total_transaction_costs.
    """
    with open(results_json, "r") as f:
        data = json.load(f)

    by_freq = data.get("averages", {}).get("by_rebalance_frequency", {})
    records = []

    for freq_str, scenarios in by_freq.items():
        try:
            freq = int(freq_str)
        except Exception:
            try:
                freq = int(str(freq_str).replace("d", ""))
            except Exception:
                continue

        for scenario, metrics in scenarios.items():
            parts = _parse_scenario_name(scenario)
            if parts is None:
                continue
            records.append(
                {
                    "scenario": scenario,
                    "hedge_type": parts.hedge_type,
                    "cost_mode": parts.cost_mode,
                    "rebalance_days": parts.rebalance_days,
                    "total_pnl": metrics.get("total_pnl", 0.0),
                    "total_pnl_pct": metrics.get("total_pnl_pct", 0.0),
                    "sharpe_ratio": metrics.get("sharpe_ratio", 0.0),
                    "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
                    "total_transaction_costs": metrics.get("total_transaction_costs", 0.0),
                }
            )

    return pd.DataFrame(records)


def plot_rebalance_metric(
    df: pd.DataFrame,
    metric: str = "total_pnl",
    title: str = "Rebalancing cadence vs metric",
    with_cost_only: bool = True,
) -> plt.Axes:
    """
    Plot metric vs rebalance_days, grouped by hedge_type and cost_mode.
    """
    if df.empty:
        raise ValueError("No data to plot")

    plot_df = df.copy()
    if with_cost_only:
        plot_df = plot_df[plot_df["cost_mode"] == "with_cost"]
        if plot_df.empty:
            raise ValueError("No with_cost data available; rerun with --include-no-cost")

    sns.set_theme(style="whitegrid")
    ax = plt.gca()

    ax.figure.set_size_inches(10, 6)

    plot_df = plot_df.assign(label=plot_df["hedge_type"] + " | " + plot_df["cost_mode"])
    hue_order = sorted(plot_df["label"].unique())
    style_order = sorted(plot_df["hedge_type"].unique())

    sns.lineplot(
        data=plot_df,
        x="rebalance_days",
        y=metric,
        hue="label",
        style="hedge_type",
        marker="o",
        ax=ax,
        hue_order=hue_order,
        style_order=style_order,
    )

    ax.set_title(title)
    ax.set_xlabel("Rebalance frequency (days)")
    ax.set_ylabel(metric.replace("_", " ").title())
    handles, labels = ax.get_legend_handles_labels()
    filtered = [
        (h, l)
        for h, l in zip(handles, labels)
        if l not in {"delta", "delta_vega", "hedge_type", "label"}
    ]
    if filtered:
        handles, labels = zip(*filtered)
        ax.legend(handles, labels, title="Hedge / Cost", frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    else:
        ax.legend().remove()
    ax.figure.tight_layout()
    return ax
