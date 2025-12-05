"""
Data loading utilities for options hedging simulation.

Handles loading CSV data, computing mid prices, and providing helper functions
for finding ATM strikes and hedge options.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import date, datetime
import yaml


def load_config(config_path: str = "config/settings.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the settings.yaml file
        
    Returns:
        Dictionary containing all configuration settings
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert maturity_dates strings to date objects if they're strings
    if 'maturity_dates' in config:
        config['maturity_dates'] = [
            pd.to_datetime(d).date() if isinstance(d, str) else d
            for d in config['maturity_dates']
        ]
    
    return config


def load_options_data(config: Dict, base_path: str = ".") -> pd.DataFrame:
    """
    Load options data from CSV and compute mid prices.
    
    Args:
        config: Configuration dictionary with data_storage settings
        base_path: Base path for relative file paths
        
    Returns:
        DataFrame with parsed dates and computed mid prices
    """
    # Build file path
    data_dir = config['data_storage']['directory']
    filename = config['data_storage']['filename']
    file_path = Path(base_path) / data_dir / filename
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date']).dt.date
    df['expiry'] = pd.to_datetime(df['expiry']).dt.date
    
    # Compute mid prices: (bid + ask) / 2
    df['call_mid'] = (df['call_bid'] + df['call_ask']) / 2
    df['put_mid'] = (df['put_bid'] + df['put_ask']) / 2
    
    # Sort by ticker, expiry, date, strike for consistent ordering
    df = df.sort_values(['ticker', 'expiry', 'date', 'strike']).reset_index(drop=True)
    
    return df


def get_atm_strike(df: pd.DataFrame, ticker: str, trading_date: date, 
                   expiry: date) -> float:
    """
    Find the ATM strike (closest to underlying price) for given ticker/date/expiry.
    
    Args:
        df: Options DataFrame
        ticker: Ticker symbol (e.g., 'AAPL.O')
        trading_date: The trading date
        expiry: The option expiry date
        
    Returns:
        Strike price closest to the underlying close price
    """
    mask = (df['ticker'] == ticker) & (df['date'] == trading_date) & (df['expiry'] == expiry)
    subset = df[mask]
    
    if subset.empty:
        raise ValueError(f"No data for {ticker} on {trading_date} expiry {expiry}")
    
    underlying_price = subset['underlying_close'].iloc[0]
    
    # Find strike closest to underlying price
    strikes = subset['strike'].values
    atm_idx = np.argmin(np.abs(strikes - underlying_price))
    
    return strikes[atm_idx]


def get_next_expiry(current_expiry: date, maturity_dates: List[date]) -> Optional[date]:
    """
    Get the next expiry date after the current one.
    
    Args:
        current_expiry: The current option expiry date
        maturity_dates: List of all available expiry dates (sorted)
        
    Returns:
        The next expiry date, or None if current is the last
    """
    # Ensure sorted
    sorted_dates = sorted(maturity_dates)
    
    for exp_date in sorted_dates:
        if exp_date > current_expiry:
            return exp_date
    
    return None


def get_previous_expiry(current_expiry: date, maturity_dates: List[date]) -> Optional[date]:
    """
    Get the previous expiry date before the current one.
    
    Args:
        current_expiry: The current option expiry date
        maturity_dates: List of all available expiry dates
        
    Returns:
        The previous expiry date, or None if current is the first
    """
    sorted_dates = sorted(maturity_dates, reverse=True)
    
    for exp_date in sorted_dates:
        if exp_date < current_expiry:
            return exp_date
    
    return None


def filter_data(df: pd.DataFrame, ticker: str, trading_date: date, 
                expiry: date, strike: float) -> pd.Series:
    """
    Get a single row matching all criteria.
    
    Args:
        df: Options DataFrame
        ticker: Ticker symbol
        trading_date: Trading date
        expiry: Option expiry date
        strike: Strike price
        
    Returns:
        Series containing the matching row data
    """
    mask = (
        (df['ticker'] == ticker) & 
        (df['date'] == trading_date) & 
        (df['expiry'] == expiry) & 
        (df['strike'] == strike)
    )
    subset = df[mask]
    
    if subset.empty:
        raise ValueError(f"No data for {ticker} {trading_date} {expiry} strike {strike}")
    
    return subset.iloc[0]


def get_available_dates(df: pd.DataFrame, ticker: str, expiry: date) -> List[date]:
    """
    Get all trading dates available for a ticker/expiry combination.
    
    Args:
        df: Options DataFrame
        ticker: Ticker symbol
        expiry: Option expiry date
        
    Returns:
        Sorted list of available trading dates
    """
    mask = (df['ticker'] == ticker) & (df['expiry'] == expiry)
    dates = df[mask]['date'].unique()
    return sorted(dates)


def get_hedge_option_data(df: pd.DataFrame, ticker: str, trading_date: date,
                          hedge_expiry: date) -> Optional[Dict]:
    """
    Get ATM option data for use as a hedging instrument.
    
    Args:
        df: Options DataFrame
        ticker: Ticker symbol
        trading_date: The trading date
        hedge_expiry: The expiry date of the hedge option
        
    Returns:
        Dictionary with hedge option Greeks and prices, or None if not available
    """
    try:
        atm_strike = get_atm_strike(df, ticker, trading_date, hedge_expiry)
        row = filter_data(df, ticker, trading_date, hedge_expiry, atm_strike)
        
        # Use call option for hedging (convention)
        return {
            'ticker': ticker,
            'expiry': hedge_expiry,
            'strike': atm_strike,
            'option_type': 'call',
            'delta': row['call_delta'],
            'gamma': row['call_gamma'],
            'vega': row['call_vega'],
            'theta': row['call_theta'],
            'mid_price': row['call_mid'],
            'underlying_price': row['underlying_close']
        }
    except (ValueError, IndexError):
        return None


def get_straddle_cost(df: pd.DataFrame, ticker: str, expiry: date) -> Dict:
    """
    Get the cost of one ATM straddle at the start of the expiry period.
    
    Args:
        df: Options DataFrame
        ticker: Ticker symbol
        expiry: Option expiry date
        
    Returns:
        Dictionary with straddle details including cost per contract
    """
    # Get first available date for this expiry
    available_dates = get_available_dates(df, ticker, expiry)
    
    if not available_dates:
        raise ValueError(f"No data available for {ticker} expiry {expiry}")
    
    first_date = available_dates[0]
    atm_strike = get_atm_strike(df, ticker, first_date, expiry)
    row = filter_data(df, ticker, first_date, atm_strike, expiry)
    
    # Straddle = 1 call + 1 put at same strike
    call_mid = row['call_mid']
    put_mid = row['put_mid']
    straddle_premium = call_mid + put_mid  # Per share
    straddle_cost = straddle_premium * 100  # Per contract (100 shares)
    
    return {
        'ticker': ticker,
        'expiry': expiry,
        'strike': atm_strike,
        'first_date': first_date,
        'call_mid': call_mid,
        'put_mid': put_mid,
        'straddle_premium': straddle_premium,
        'straddle_cost_per_contract': straddle_cost,
        'underlying_price': row['underlying_close']
    }


def get_underlying_price(df: pd.DataFrame, ticker: str, trading_date: date) -> float:
    """
    Get the underlying stock price for a ticker on a given date.
    
    Args:
        df: Options DataFrame
        ticker: Ticker symbol
        trading_date: Trading date
        
    Returns:
        Underlying close price
    """
    mask = (df['ticker'] == ticker) & (df['date'] == trading_date)
    subset = df[mask]
    
    if subset.empty:
        raise ValueError(f"No data for {ticker} on {trading_date}")
    
    return subset['underlying_close'].iloc[0]


def get_available_tickers(df: pd.DataFrame) -> List[str]:
    """Get list of unique tickers in the dataset."""
    return sorted(df['ticker'].unique().tolist())


def get_available_expiries(df: pd.DataFrame, ticker: Optional[str] = None) -> List[date]:
    """
    Get list of unique expiry dates, optionally filtered by ticker.
    
    Args:
        df: Options DataFrame
        ticker: Optional ticker to filter by
        
    Returns:
        Sorted list of expiry dates
    """
    if ticker:
        mask = df['ticker'] == ticker
        expiries = df[mask]['expiry'].unique()
    else:
        expiries = df['expiry'].unique()
    
    return sorted(expiries)
