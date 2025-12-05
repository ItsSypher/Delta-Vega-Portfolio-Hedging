#!/usr/bin/env python3
"""
Hedgesimple - Options Hedging Simulation

Main entry point for running delta and delta-vega hedging simulations
with transaction costs and multiple rebalancing frequencies.

Usage:
    python main.py [--config CONFIG_PATH]

Output:
    - results/hedging_timeseries.csv - Daily time-series data
    - results/hedging_results.json   - Complete results with averages
"""

import argparse
from pathlib import Path
from datetime import datetime

from src.data_loader import load_config, load_options_data
from src.simulator import HedgeSimulator, save_results_csv, save_results_json


def apply_cli_overrides(config: dict, args) -> dict:
    """Apply command-line argument overrides to configuration."""
    
    # Initialize subset if not present
    if 'subset' not in config:
        config['subset'] = {}
    
    subset = config['subset']
    
    # Quick mode: minimal simulation for fast testing
    if args.quick:
        subset['enabled'] = True
        subset['max_tickers'] = 1
        subset['max_expiries'] = 2
        subset['rebalance_frequencies'] = [1]
        print("\n*** QUICK MODE: 1 ticker, 2 expiries, daily rebalance ***")
    
    # Specific tickers
    if args.tickers:
        subset['enabled'] = True
        subset['tickers'] = args.tickers
    
    # Max tickers
    if args.max_tickers:
        subset['enabled'] = True
        subset['max_tickers'] = args.max_tickers
    
    # Max expiries
    if args.max_expiries:
        subset['enabled'] = True
        subset['max_expiries'] = args.max_expiries
    
    # Rebalance frequencies
    if args.rebalance:
        subset['enabled'] = True
        subset['rebalance_frequencies'] = args.rebalance
    
    # Delta only
    if args.delta_only:
        subset['enabled'] = True
        subset['hedge_types'] = ['delta']
    
    # No cost only
    if args.no_cost:
        subset['enabled'] = True
        subset['cost_modes'] = ['no_cost']
    
    # Parallel execution overrides
    if 'parallel' not in config:
        config['parallel'] = {}
    
    if args.no_parallel:
        config['parallel']['enabled'] = False
    
    if args.workers:
        config['parallel']['max_workers'] = args.workers
    
    return config


def print_summary(results: dict):
    """Print a formatted summary of simulation results."""
    
    print("\n" + "=" * 70)
    print("HEDGING SIMULATION RESULTS")
    print("=" * 70)
    
    # Metadata
    metadata = results.get('metadata', {})
    print(f"\nConfiguration:")
    print(f"  Initial Capital:     ${metadata.get('initial_capital', 10000):,.0f}")
    print(f"  Transaction Cost:    {metadata.get('transaction_cost_pct', 5.0)}%")
    print(f"  Tickers:             {metadata.get('num_tickers', 0)}")
    print(f"  Expiries:            {metadata.get('num_expiries', 0)}")
    print(f"  Scenarios:           {metadata.get('num_scenarios', 0)}")
    print(f"  Rebalance Freqs:     {metadata.get('rebalance_frequencies', [])}")
    
    # Global averages
    global_avg = results.get('averages', {}).get('global', {})
    
    if global_avg:
        print("\n" + "-" * 70)
        print("GLOBAL AVERAGE RESULTS (across all tickers and expiries)")
        print("-" * 70)
        
        # Group by rebalance frequency for display
        rebal_freqs = metadata.get('rebalance_frequencies', [1, 5, 21])
        
        print(f"\n{'Scenario':<40} {'P&L':>12} {'P&L %':>10} {'Sharpe':>10} {'MaxDD %':>10}")
        print("-" * 82)
        
        for freq in rebal_freqs:
            for hedge in ['delta', 'delta_vega']:
                for cost in ['no_cost', 'with_cost']:
                    scenario = f"{hedge}_{cost}_rebal_{freq}d"
                    if scenario in global_avg:
                        data = global_avg[scenario]
                        pnl = data.get('total_pnl', 0)
                        pnl_pct = data.get('total_pnl_pct', 0)
                        sharpe = data.get('sharpe_ratio', 0)
                        maxdd = data.get('max_drawdown_pct', 0)
                        
                        print(f"{scenario:<40} ${pnl:>10,.0f} {pnl_pct:>9.2f}% {sharpe:>10.3f} {maxdd:>9.2f}%")
            print()  # Blank line between frequencies
    
    # Comparisons
    comparisons = results.get('comparisons', {})
    
    if comparisons:
        print("-" * 70)
        print("KEY COMPARISONS")
        print("-" * 70)
        
        # Delta vs Delta-Vega
        dv_comp = comparisons.get('delta_vs_delta_vega', {})
        print(f"\nDelta vs Delta-Vega (daily rebalance, no costs):")
        print(f"  P&L Difference:      ${dv_comp.get('pnl_difference', 0):,.2f}")
        print(f"  Sharpe Difference:   {dv_comp.get('sharpe_difference', 0):.3f}")
        
        # Cost Impact
        cost_impact = comparisons.get('cost_impact', {})
        delta_impact = cost_impact.get('delta', {})
        dv_impact = cost_impact.get('delta_vega', {})
        
        print(f"\nTransaction Cost Impact (5% cost):")
        print(f"  Delta P&L Reduction:       ${delta_impact.get('pnl_reduction', 0):,.2f} ({delta_impact.get('pnl_reduction_pct', 0):.1f}%)")
        print(f"  Delta-Vega P&L Reduction:  ${dv_impact.get('pnl_reduction', 0):,.2f} ({dv_impact.get('pnl_reduction_pct', 0):.1f}%)")
        
        # Optimal rebalancing
        rebal_analysis = comparisons.get('rebalance_analysis', {})
        print(f"\nOptimal Rebalancing Frequency (with costs):")
        print(f"  Delta:       Every {rebal_analysis.get('optimal_delta', 1)} days")
        print(f"  Delta-Vega:  Every {rebal_analysis.get('optimal_delta_vega', 1)} days")
    
    # Per-ticker highlights
    by_ticker = results.get('averages', {}).get('by_ticker', {})
    
    if by_ticker:
        print("\n" + "-" * 70)
        print("PER-TICKER RESULTS (delta_vega_with_cost_rebal_1d)")
        print("-" * 70)
        print(f"\n{'Ticker':<12} {'Avg P&L':>12} {'Sharpe':>10}")
        print("-" * 34)
        
        for ticker, data in sorted(by_ticker.items()):
            scenario_data = data.get('delta_vega_with_cost_rebal_1d', {})
            pnl = scenario_data.get('total_pnl', 0)
            sharpe = scenario_data.get('sharpe_ratio', 0)
            print(f"{ticker:<12} ${pnl:>10,.0f} {sharpe:>10.3f}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description='Run options hedging simulation'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode: 1 ticker, 2 expiries, daily rebalance only'
    )
    parser.add_argument(
        '--tickers', '-t',
        type=str,
        nargs='+',
        help='Specific tickers to run (e.g., AAPL.O TSLA.O)'
    )
    parser.add_argument(
        '--max-tickers',
        type=int,
        help='Maximum number of tickers to run'
    )
    parser.add_argument(
        '--max-expiries',
        type=int,
        help='Maximum number of expiries to run'
    )
    parser.add_argument(
        '--rebalance', '-r',
        type=int,
        nargs='+',
        help='Rebalance frequencies to run (e.g., 1 5)'
    )
    parser.add_argument(
        '--delta-only',
        action='store_true',
        help='Run only delta hedging (skip delta-vega)'
    )
    parser.add_argument(
        '--no-cost',
        action='store_true', 
        help='Run only no-cost scenarios'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel execution (run sequentially)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of parallel workers'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("HEDGESIMPLE - Options Hedging Simulation")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Apply CLI overrides to config
    config = apply_cli_overrides(config, args)
    
    # Print subset info if active
    subset = config.get('subset', {})
    if subset.get('enabled'):
        print("\n*** SUBSET MODE ENABLED ***")
        if subset.get('tickers'):
            print(f"  Tickers: {subset['tickers']}")
        elif subset.get('max_tickers'):
            print(f"  Max tickers: {subset['max_tickers']}")
        if subset.get('expiries'):
            print(f"  Expiries: {subset['expiries']}")
        elif subset.get('max_expiries'):
            print(f"  Max expiries: {subset['max_expiries']}")
        if subset.get('hedge_types'):
            print(f"  Hedge types: {subset['hedge_types']}")
        if subset.get('cost_modes'):
            print(f"  Cost modes: {subset['cost_modes']}")
        if subset.get('rebalance_frequencies'):
            print(f"  Rebalance frequencies: {subset['rebalance_frequencies']}")
    
    # Load data
    print("\nLoading options data...")
    data = load_options_data(config)
    print(f"  Loaded {len(data):,} rows")
    print(f"  Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  Tickers: {sorted(data['ticker'].unique())}")
    
    # Initialize simulator
    print("\nInitializing simulator...")
    simulator = HedgeSimulator(config, data)
    
    # Run simulation
    print("\nRunning simulations...")
    results = simulator.run_all()
    
    # Create output directory
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('directory', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save outputs
    csv_path = output_dir / output_config.get('csv_filename', 'hedging_timeseries.csv')
    json_path = output_dir / output_config.get('json_filename', 'hedging_results.json')
    
    print("\nSaving results...")
    save_results_csv(results, str(csv_path))
    save_results_json(results, str(json_path))
    
    # Print summary
    print_summary(results)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput files:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")


if __name__ == '__main__':
    main()
