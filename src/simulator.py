"""
Hedging simulation engine.

Runs delta and delta-vega hedging simulations across multiple tickers,
expiries, rebalancing frequencies, and cost scenarios.

Produces comprehensive results with per-ticker, per-expiry, and global averages.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import date
from dataclasses import dataclass, field
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import os

from .data_loader import (
    get_available_dates,
    get_atm_strike,
    filter_data,
    get_next_expiry,
    get_hedge_option_data,
    get_underlying_price
)
from .portfolio import Portfolio, PortfolioFactory
from .hedger import Hedger


@dataclass
class DailySnapshot:
    """Snapshot of simulation state for one day."""
    date: date
    dte: int
    underlying_price: float
    portfolio_value: float
    portfolio_delta: float
    portfolio_vega: float
    portfolio_gamma: float
    portfolio_theta: float
    scenarios: Dict  # Scenario name -> scenario data
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'dte': self.dte,
            'underlying_price': self.underlying_price,
            'portfolio_value': self.portfolio_value,
            'portfolio_delta': self.portfolio_delta,
            'portfolio_vega': self.portfolio_vega,
            'portfolio_gamma': self.portfolio_gamma,
            'portfolio_theta': self.portfolio_theta,
            'scenarios': self.scenarios
        }


class HedgeSimulator:
    """
    Main simulation engine for hedging strategies.
    
    Compares:
    - Delta vs Delta-Vega hedging
    - With vs Without transaction costs
    - Multiple rebalancing frequencies
    """
    
    def __init__(self, config: Dict, data: pd.DataFrame):
        """
        Initialize simulator.
        
        Args:
            config: Configuration dictionary
            data: Options DataFrame with mid prices computed
        """
        self.config = config
        self.data = data
        
        # Extract settings
        hedging_config = config.get('hedging', {})
        self.transaction_cost_pct = hedging_config.get('transaction_cost_pct', 5.0)
        self.rebalance_frequencies = hedging_config.get('rebalance_frequencies', [1, 5, 21])
        self.maturity_dates = config.get('maturity_dates', [])
        
        # Initialize hedger
        self.hedger = Hedger(transaction_cost_pct=self.transaction_cost_pct)
        
        # Build scenario list
        self.scenarios = self._build_scenario_list()
    
    def _build_scenario_list(self) -> List[str]:
        """
        Build list of all scenario names.
        
        Format: {hedge_type}_{cost_mode}_rebal_{freq}d
        Example: delta_vega_with_cost_rebal_5d
        """
        scenarios = []
        for hedge_type in ['delta', 'delta_vega']:
            for cost_mode in ['no_cost', 'with_cost']:
                for freq in self.rebalance_frequencies:
                    scenario_name = f"{hedge_type}_{cost_mode}_rebal_{freq}d"
                    scenarios.append(scenario_name)
        return scenarios
    
    def should_rebalance(self, day_index: int, rebalance_freq: int) -> bool:
        """Check if this is a rebalancing day."""
        return day_index % rebalance_freq == 0
    
    def run_single_expiry(self, ticker: str, expiry: date) -> Dict:
        """
        Run hedging simulation for one ticker/expiry combination.
        
        Args:
            ticker: Ticker symbol
            expiry: Option expiry date
            
        Returns:
            Dictionary with daily snapshots and summary statistics
        """
        # Get available trading dates
        trading_dates = get_available_dates(self.data, ticker, expiry)
        
        if len(trading_dates) < 2:
            return {'error': f"Insufficient data for {ticker} {expiry}"}
        
        # Create portfolio (ATM straddle sized to initial capital)
        try:
            portfolio = PortfolioFactory.from_config(
                self.config, self.data, ticker, expiry
            )
        except ValueError as e:
            return {'error': str(e)}
        
        # Get hedge option expiry (next monthly)
        hedge_expiry = get_next_expiry(expiry, self.maturity_dates)
        
        # Initialize scenario state tracking
        scenario_state = {}
        for scenario in self.scenarios:
            scenario_state[scenario] = {
                'prev_shares': 0.0,
                'prev_options': 0.0,
                'prev_underlying_price': None,
                'prev_option_price': None,
                'cumulative_pnl': 0.0,
                'cumulative_costs': 0.0,
                'daily_pnls': [],
                'peak_value': 0.0
            }
        
        # Store daily snapshots
        daily_snapshots = []
        initial_portfolio_value = None
        
        # Run simulation day by day
        for day_idx, trading_date in enumerate(trading_dates):
            # Get portfolio Greeks
            try:
                greeks = portfolio.calculate_greeks(self.data, trading_date)
            except ValueError:
                continue
            
            portfolio_value = greeks['value']
            portfolio_delta = greeks['delta']
            portfolio_vega = greeks['vega']
            portfolio_gamma = greeks['gamma']
            portfolio_theta = greeks['theta']
            
            if initial_portfolio_value is None:
                initial_portfolio_value = portfolio_value
            
            # Get underlying price
            try:
                underlying_price = get_underlying_price(self.data, ticker, trading_date)
            except ValueError:
                continue
            
            # Get hedge option data (for delta-vega scenarios)
            hedge_option = None
            if hedge_expiry:
                hedge_option = get_hedge_option_data(
                    self.data, ticker, trading_date, hedge_expiry
                )
            
            # Get DTE
            try:
                row = filter_data(self.data, ticker, trading_date, expiry, 
                                  portfolio.positions[0].strike)
                dte = int(row['dte'])
            except (ValueError, IndexError):
                dte = (expiry - trading_date).days
            
            # Process each scenario
            scenario_results = {}
            
            for scenario in self.scenarios:
                # Parse scenario name
                parts = scenario.split('_')
                if 'delta_vega' in scenario:
                    hedge_type = 'delta_vega'
                    cost_mode = 'with_cost' if 'with_cost' in scenario else 'no_cost'
                else:
                    hedge_type = 'delta'
                    cost_mode = 'with_cost' if 'with_cost' in scenario else 'no_cost'
                
                # Extract rebalance frequency
                rebal_idx = scenario.rfind('rebal_')
                freq_str = scenario[rebal_idx:].replace('rebal_', '').replace('d', '')
                rebalance_freq = int(freq_str)
                
                with_costs = (cost_mode == 'with_cost')
                state = scenario_state[scenario]
                
                # Calculate P&L from previous day's hedge
                daily_pnl = 0.0
                hedge_pnl = 0.0
                
                if day_idx > 0 and state['prev_underlying_price'] is not None:
                    # Share hedge P&L
                    share_pnl = state['prev_shares'] * (
                        underlying_price - state['prev_underlying_price']
                    )
                    
                    # Option hedge P&L (for delta-vega)
                    option_pnl = 0.0
                    if hedge_type == 'delta_vega' and hedge_option and state['prev_option_price']:
                        option_pnl = state['prev_options'] * (
                            hedge_option['mid_price'] - state['prev_option_price']
                        ) * 100
                    
                    hedge_pnl = share_pnl + option_pnl
                
                # Portfolio P&L (change in portfolio value)
                if day_idx > 0:
                    portfolio_pnl = portfolio_value - scenario_state[scenario].get(
                        'prev_portfolio_value', portfolio_value
                    )
                else:
                    portfolio_pnl = 0.0
                
                # Total daily P&L = portfolio change + hedge P&L
                daily_pnl = portfolio_pnl + hedge_pnl
                
                # Check if rebalance day
                transaction_cost = 0.0
                
                if self.should_rebalance(day_idx, rebalance_freq):
                    if hedge_type == 'delta':
                        result = self.hedger.hedge_delta(
                            portfolio_delta=portfolio_delta,
                            underlying_price=underlying_price,
                            prev_shares=state['prev_shares'],
                            with_costs=with_costs
                        )
                        new_shares = result.shares
                        new_options = 0.0
                        transaction_cost = result.transaction_cost
                        
                    else:  # delta_vega
                        if hedge_option:
                            result = self.hedger.hedge_delta_vega(
                                portfolio_delta=portfolio_delta,
                                portfolio_vega=portfolio_vega,
                                hedge_option=hedge_option,
                                underlying_price=underlying_price,
                                prev_shares=state['prev_shares'],
                                prev_options=state['prev_options'],
                                with_costs=with_costs
                            )
                            new_shares = result.shares
                            new_options = result.options
                            transaction_cost = result.transaction_cost
                        else:
                            # No hedge option available, fall back to delta
                            result = self.hedger.hedge_delta(
                                portfolio_delta=portfolio_delta,
                                underlying_price=underlying_price,
                                prev_shares=state['prev_shares'],
                                with_costs=with_costs
                            )
                            new_shares = result.shares
                            new_options = 0.0
                            transaction_cost = result.transaction_cost
                else:
                    # No rebalance - carry forward positions
                    new_shares = state['prev_shares']
                    new_options = state['prev_options']
                
                # Net daily P&L after costs
                net_daily_pnl = daily_pnl - transaction_cost
                
                # Update cumulative metrics
                state['cumulative_pnl'] += net_daily_pnl
                state['cumulative_costs'] += transaction_cost
                state['daily_pnls'].append(net_daily_pnl)
                
                # Track peak for drawdown
                total_value = initial_portfolio_value + state['cumulative_pnl']
                if total_value > state['peak_value']:
                    state['peak_value'] = total_value
                
                # Update state for next day
                state['prev_shares'] = new_shares
                state['prev_options'] = new_options
                state['prev_underlying_price'] = underlying_price
                state['prev_portfolio_value'] = portfolio_value
                if hedge_option:
                    state['prev_option_price'] = hedge_option['mid_price']
                
                # Record scenario results
                scenario_results[scenario] = {
                    'shares': new_shares,
                    'options': new_options,
                    'daily_pnl': net_daily_pnl,
                    'hedge_pnl': hedge_pnl,
                    'transaction_cost': transaction_cost,
                    'cumulative_pnl': state['cumulative_pnl'],
                    'cumulative_costs': state['cumulative_costs']
                }
            
            # Create daily snapshot
            snapshot = DailySnapshot(
                date=trading_date,
                dte=dte,
                underlying_price=underlying_price,
                portfolio_value=portfolio_value,
                portfolio_delta=portfolio_delta,
                portfolio_vega=portfolio_vega,
                portfolio_gamma=portfolio_gamma,
                portfolio_theta=portfolio_theta,
                scenarios=scenario_results
            )
            daily_snapshots.append(snapshot)
        
        # Calculate summary statistics
        summary = self._calculate_summary(
            scenario_state, initial_portfolio_value, len(trading_dates)
        )
        
        return {
            'ticker': ticker,
            'expiry': expiry.isoformat(),
            'initial_portfolio_value': initial_portfolio_value,
            'num_trading_days': len(trading_dates),
            'portfolio_config': portfolio.to_dict(),
            'hedge_expiry': hedge_expiry.isoformat() if hedge_expiry else None,
            'daily_snapshots': [s.to_dict() for s in daily_snapshots],
            'summary': summary
        }
    
    def _calculate_summary(self, scenario_state: Dict, 
                           initial_value: float, num_days: int) -> Dict:
        """Calculate summary statistics for all scenarios."""
        summary = {}
        
        for scenario, state in scenario_state.items():
            daily_pnls = np.array(state['daily_pnls'])
            
            if len(daily_pnls) == 0:
                continue
            
            # Filter out any NaN values
            daily_pnls = daily_pnls[np.isfinite(daily_pnls)]
            
            if len(daily_pnls) == 0:
                continue
            
            total_pnl = state['cumulative_pnl']
            total_costs = state['cumulative_costs']
            
            # Handle NaN in cumulative values
            if not np.isfinite(total_pnl):
                total_pnl = float(np.nansum(state['daily_pnls']))
            if not np.isfinite(total_costs):
                total_costs = 0.0
            
            # Basic stats
            mean_daily_pnl = float(np.nanmean(daily_pnls))
            std_daily_pnl = float(np.nanstd(daily_pnls)) if len(daily_pnls) > 1 else 0.0
            
            # Sharpe ratio (annualized, assuming 252 trading days)
            if std_daily_pnl > 0 and np.isfinite(std_daily_pnl):
                sharpe = (mean_daily_pnl / std_daily_pnl) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            # Max drawdown
            cumulative = np.cumsum(daily_pnls)
            running_max = np.maximum.accumulate(cumulative + initial_value)
            drawdowns = (running_max - (cumulative + initial_value)) / running_max
            max_drawdown = float(np.nanmax(drawdowns)) if len(drawdowns) > 0 else 0.0
            
            # Final values
            final_value = initial_value + total_pnl
            pnl_pct = (total_pnl / initial_value * 100) if initial_value else 0.0
            
            summary[scenario] = {
                'total_pnl': float(total_pnl),
                'total_pnl_pct': float(pnl_pct),
                'final_value': float(final_value),
                'daily_pnl_mean': float(mean_daily_pnl),
                'daily_pnl_std': float(std_daily_pnl),
                'sharpe_ratio': float(sharpe),
                'max_drawdown': float(max_drawdown),
                'max_drawdown_pct': float(max_drawdown * 100),
                'total_transaction_costs': float(total_costs),
                'cost_as_pct_of_initial': float((total_costs / initial_value * 100) if initial_value else 0),
                'num_trading_days': num_days
            }
        
        return summary
    
    def _apply_subset_filter(self, tickers: List, expiries: List) -> Tuple[List, List]:
        """Apply subset configuration to filter tickers and expiries."""
        subset = self.config.get('subset', {})
        
        if not subset.get('enabled', False):
            return tickers, expiries
        
        # Filter tickers
        if subset.get('tickers'):
            # Specific tickers list
            tickers = [t for t in tickers if t in subset['tickers']]
        elif subset.get('max_tickers'):
            # Limit number of tickers
            tickers = tickers[:subset['max_tickers']]
        
        # Filter expiries
        if subset.get('expiries'):
            # Specific expiries list (convert strings to date objects for comparison)
            from datetime import datetime
            subset_expiries = []
            for e in subset['expiries']:
                if isinstance(e, str):
                    subset_expiries.append(datetime.strptime(e, '%Y-%m-%d').date())
                else:
                    subset_expiries.append(e)
            expiries = [e for e in expiries if e in subset_expiries]
        elif subset.get('max_expiries'):
            # Limit number of expiries
            expiries = expiries[:subset['max_expiries']]
        
        return tickers, expiries
    
    def _apply_scenario_filter(self) -> List[str]:
        """Apply subset configuration to filter scenarios."""
        subset = self.config.get('subset', {})
        
        if not subset.get('enabled', False):
            return self.scenarios
        
        # Get filters
        hedge_types = subset.get('hedge_types') or ['delta', 'delta_vega']
        cost_modes = subset.get('cost_modes') or ['no_cost', 'with_cost']
        rebal_freqs = subset.get('rebalance_frequencies') or self.rebalance_frequencies
        
        # Rebuild scenarios with filters
        scenarios = []
        for hedge_type in hedge_types:
            for cost_mode in cost_modes:
                for freq in rebal_freqs:
                    if freq in self.rebalance_frequencies:  # Must be in original list
                        scenario_name = f"{hedge_type}_{cost_mode}_rebal_{freq}d"
                        if scenario_name in self.scenarios:
                            scenarios.append(scenario_name)
        
        return scenarios
    
    def run_all(self) -> Dict:
        """
        Run simulation for all ticker/expiry combinations.
        
        Supports parallel execution via configuration.
        
        Returns:
            Comprehensive results dictionary with all simulations and averages
        """
        tickers = self.config.get('tickers', [])
        expiries = self.maturity_dates
        
        # Apply subset filtering
        tickers, expiries = self._apply_subset_filter(tickers, expiries)
        
        # Filter scenarios
        active_scenarios = self._apply_scenario_filter()
        
        # Parallelization settings
        parallel_config = self.config.get('parallel', {})
        parallel_enabled = parallel_config.get('enabled', True)
        max_workers = parallel_config.get('max_workers', None)
        executor_type = parallel_config.get('executor', 'thread')  # 'thread' or 'process'
        
        # Default max_workers to CPU count
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), 8)
        
        results_by_ticker = {ticker: {} for ticker in tickers}
        
        total_sims = len(tickers) * len(expiries)
        
        print(f"Running {total_sims} simulations ({len(tickers)} tickers × {len(expiries)} expiries)...")
        print(f"Active scenarios: {len(active_scenarios)}")
        
        if parallel_enabled and total_sims > 1:
            print(f"Parallel execution: {executor_type} pool with {max_workers} workers")
            results_by_ticker = self._run_parallel(
                tickers, expiries, max_workers, executor_type
            )
        else:
            print("Sequential execution")
            results_by_ticker = self._run_sequential(tickers, expiries)
        
        # Count successful simulations
        completed = sum(
            len(expiry_results) 
            for expiry_results in results_by_ticker.values()
        )
        print(f"Completed {completed} simulations")
        
        # Compute averages
        averages = self._compute_averages(results_by_ticker)
        
        # Compute comparisons
        comparisons = self._compute_comparisons(averages)
        
        # Determine active rebalance frequencies
        subset = self.config.get('subset', {})
        active_rebal_freqs = subset.get('rebalance_frequencies') if subset.get('enabled') else None
        active_rebal_freqs = active_rebal_freqs or self.rebalance_frequencies
        
        return {
            'metadata': {
                'num_tickers': len(tickers),
                'num_expiries': len(expiries),
                'num_scenarios': len(active_scenarios),
                'rebalance_frequencies': active_rebal_freqs,
                'transaction_cost_pct': self.transaction_cost_pct,
                'initial_capital': self.config.get('portfolio', {}).get('initial_capital', 10000),
                'subset_mode': subset.get('enabled', False),
                'tickers': tickers,
                'expiries': [e.isoformat() for e in expiries]
            },
            'config': {
                'portfolio': self.config.get('portfolio', {}),
                'hedging': self.config.get('hedging', {})
            },
            'results_by_ticker': results_by_ticker,
            'averages': averages,
            'comparisons': comparisons,
            'scenarios': active_scenarios
        }
    
    def _run_sequential(self, tickers: List[str], expiries: List[date]) -> Dict:
        """Run simulations sequentially."""
        results_by_ticker = {ticker: {} for ticker in tickers}
        total = len(tickers) * len(expiries)
        completed = 0
        
        for ticker in tickers:
            for expiry in expiries:
                result = self.run_single_expiry(ticker, expiry)
                
                if 'error' not in result:
                    results_by_ticker[ticker][expiry.isoformat()] = result
                
                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total} simulations")
        
        return results_by_ticker
    
    def _run_parallel(self, tickers: List[str], expiries: List[date], 
                      max_workers: int, executor_type: str) -> Dict:
        """Run simulations in parallel using thread or process pool."""
        results_by_ticker = {ticker: {} for ticker in tickers}
        
        # Create list of all (ticker, expiry) jobs
        jobs = [(ticker, expiry) for ticker in tickers for expiry in expiries]
        total = len(jobs)
        completed = 0
        
        # Choose executor type
        # Note: ThreadPoolExecutor is usually better for I/O bound tasks
        # and works well here since we're doing numpy calculations
        ExecutorClass = ThreadPoolExecutor if executor_type == 'thread' else ProcessPoolExecutor
        
        with ExecutorClass(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self._run_single_job, ticker, expiry): (ticker, expiry)
                for ticker, expiry in jobs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_job):
                ticker, expiry = future_to_job[future]
                
                try:
                    result = future.result()
                    if result is not None and 'error' not in result:
                        results_by_ticker[ticker][expiry.isoformat()] = result
                except Exception as e:
                    print(f"  Error in {ticker}/{expiry}: {e}")
                
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(f"  Progress: {completed}/{total} simulations")
        
        return results_by_ticker
    
    def _run_single_job(self, ticker: str, expiry: date) -> Optional[Dict]:
        """Wrapper for running a single simulation job (for parallel execution)."""
        try:
            return self.run_single_expiry(ticker, expiry)
        except Exception as e:
            return {'error': str(e)}
    
    def _compute_averages(self, results_by_ticker: Dict) -> Dict:
        """Compute per-ticker, per-expiry, per-frequency, and global averages."""
        
        # Collect all summaries
        all_summaries = []
        by_ticker_summaries = {}
        by_expiry_summaries = {}
        by_frequency_summaries = {}
        
        for ticker, expiry_results in results_by_ticker.items():
            if ticker not in by_ticker_summaries:
                by_ticker_summaries[ticker] = []
            
            for expiry, result in expiry_results.items():
                if 'summary' not in result:
                    continue
                
                summary = result['summary']
                all_summaries.append(summary)
                by_ticker_summaries[ticker].append(summary)
                
                if expiry not in by_expiry_summaries:
                    by_expiry_summaries[expiry] = []
                by_expiry_summaries[expiry].append(summary)
        
        # Average function
        def average_summaries(summaries: List[Dict]) -> Dict:
            if not summaries:
                return {}
            
            averaged = {}
            for scenario in self.scenarios:
                scenario_values = [s.get(scenario, {}) for s in summaries if scenario in s]
                
                if not scenario_values:
                    continue
                
                averaged[scenario] = {}
                metrics = ['total_pnl', 'total_pnl_pct', 'sharpe_ratio', 
                           'max_drawdown_pct', 'total_transaction_costs']
                
                for metric in metrics:
                    values = [v.get(metric, 0) for v in scenario_values if metric in v]
                    values = [v for v in values if v is not None and np.isfinite(v)]
                    if values:
                        averaged[scenario][metric] = float(np.nanmean(values))
                    else:
                        averaged[scenario][metric] = 0.0
            
            return averaged
        
        # Compute averages
        averages = {
            'by_ticker': {
                ticker: average_summaries(summaries) 
                for ticker, summaries in by_ticker_summaries.items()
            },
            'by_expiry': {
                expiry: average_summaries(summaries)
                for expiry, summaries in by_expiry_summaries.items()
            },
            'global': average_summaries(all_summaries)
        }
        
        # Add per-frequency averages
        by_frequency = {}
        for freq in self.rebalance_frequencies:
            freq_scenarios = [s for s in self.scenarios if f'rebal_{freq}d' in s]
            freq_avg = {}
            
            for scenario in freq_scenarios:
                values = []
                for summary in all_summaries:
                    if scenario in summary:
                        values.append(summary[scenario])
                
                if values:
                    freq_avg[scenario] = {}
                    for metric in ['total_pnl', 'total_pnl_pct', 'sharpe_ratio', 
                                   'max_drawdown_pct', 'total_transaction_costs']:
                        metric_values = [v.get(metric, 0) for v in values]
                        metric_values = [v for v in metric_values if v is not None and np.isfinite(v)]
                        if metric_values:
                            freq_avg[scenario][metric] = float(np.nanmean(metric_values))
                        else:
                            freq_avg[scenario][metric] = 0.0
            
            by_frequency[freq] = freq_avg
        
        averages['by_rebalance_frequency'] = by_frequency
        
        return averages
    
    def _compute_comparisons(self, averages: Dict) -> Dict:
        """Compute comparison metrics between strategies."""
        
        global_avg = averages.get('global', {})
        
        # Compare delta vs delta-vega (using daily rebalance as baseline)
        delta_1d = global_avg.get('delta_no_cost_rebal_1d', {})
        delta_vega_1d = global_avg.get('delta_vega_no_cost_rebal_1d', {})
        
        delta_vs_delta_vega = {
            'pnl_difference': delta_vega_1d.get('total_pnl', 0) - delta_1d.get('total_pnl', 0),
            'sharpe_difference': delta_vega_1d.get('sharpe_ratio', 0) - delta_1d.get('sharpe_ratio', 0),
        }
        
        # Compare cost impact
        delta_no_cost = global_avg.get('delta_no_cost_rebal_1d', {})
        delta_with_cost = global_avg.get('delta_with_cost_rebal_1d', {})
        dv_no_cost = global_avg.get('delta_vega_no_cost_rebal_1d', {})
        dv_with_cost = global_avg.get('delta_vega_with_cost_rebal_1d', {})
        
        cost_impact = {
            'delta': {
                'pnl_reduction': delta_with_cost.get('total_pnl', 0) - delta_no_cost.get('total_pnl', 0),
                'pnl_reduction_pct': (
                    (delta_with_cost.get('total_pnl', 0) - delta_no_cost.get('total_pnl', 0)) 
                    / abs(delta_no_cost.get('total_pnl', 1)) * 100
                ) if delta_no_cost.get('total_pnl', 0) != 0 else 0
            },
            'delta_vega': {
                'pnl_reduction': dv_with_cost.get('total_pnl', 0) - dv_no_cost.get('total_pnl', 0),
                'pnl_reduction_pct': (
                    (dv_with_cost.get('total_pnl', 0) - dv_no_cost.get('total_pnl', 0))
                    / abs(dv_no_cost.get('total_pnl', 1)) * 100
                ) if dv_no_cost.get('total_pnl', 0) != 0 else 0
            }
        }
        
        # Find optimal rebalancing frequency
        rebalance_analysis = {'by_frequency': {}}
        
        for freq in self.rebalance_frequencies:
            freq_results = {}
            for hedge in ['delta', 'delta_vega']:
                for cost in ['no_cost', 'with_cost']:
                    scenario = f"{hedge}_{cost}_rebal_{freq}d"
                    if scenario in global_avg:
                        freq_results[f"{hedge}_{cost}"] = global_avg[scenario]
            rebalance_analysis['by_frequency'][freq] = freq_results
        
        # Find best frequency for each strategy
        best_freq = {}
        for hedge in ['delta', 'delta_vega']:
            best_pnl = float('-inf')
            best_f = self.rebalance_frequencies[0]
            for freq in self.rebalance_frequencies:
                scenario = f"{hedge}_with_cost_rebal_{freq}d"
                pnl = global_avg.get(scenario, {}).get('total_pnl', float('-inf'))
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_f = freq
            best_freq[f'optimal_{hedge}'] = best_f
        
        rebalance_analysis.update(best_freq)
        
        return {
            'delta_vs_delta_vega': delta_vs_delta_vega,
            'cost_impact': cost_impact,
            'rebalance_analysis': rebalance_analysis
        }


def save_results_csv(results: Dict, output_path: str):
    """
    Save time-series results to CSV.
    
    Args:
        results: Full results dictionary from run_all()
        output_path: Path to save CSV file
    """
    rows = []
    
    for ticker, expiry_results in results.get('results_by_ticker', {}).items():
        for expiry, result in expiry_results.items():
            for snapshot in result.get('daily_snapshots', []):
                row = {
                    'ticker': ticker,
                    'expiry': expiry,
                    'date': snapshot['date'],
                    'dte': snapshot['dte'],
                    'underlying_price': snapshot['underlying_price'],
                    'portfolio_value': snapshot['portfolio_value'],
                    'portfolio_delta': snapshot['portfolio_delta'],
                    'portfolio_vega': snapshot['portfolio_vega'],
                    'portfolio_gamma': snapshot['portfolio_gamma'],
                    'portfolio_theta': snapshot['portfolio_theta']
                }
                
                # Add scenario data
                for scenario, data in snapshot.get('scenarios', {}).items():
                    for key, value in data.items():
                        row[f"{scenario}_{key}"] = value
                
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved time-series CSV to {output_path}")


def save_results_json(results: Dict, output_path: str):
    """
    Save full results to JSON.
    
    Args:
        results: Full results dictionary from run_all()
        output_path: Path to save JSON file
    """
    # Convert any remaining date objects to strings
    def convert_dates(obj):
        if isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert_dates(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_dates(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean_results = convert_dates(results)
    
    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"Saved full results JSON to {output_path}")
