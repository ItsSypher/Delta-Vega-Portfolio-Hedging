"""
Portfolio management for options hedging simulation.

Handles portfolio construction, Greek aggregation, and mark-to-market calculations.
Supports both ATM straddle (auto-sized to initial capital) and custom positions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import date
from dataclasses import dataclass, field

from .data_loader import (
    get_atm_strike, 
    filter_data, 
    get_available_dates,
    get_straddle_cost
)


@dataclass
class Position:
    """Represents a single options position."""
    ticker: str
    expiry: date
    strike: float
    option_type: str  # 'call' or 'put'
    quantity: float   # Can be fractional
    
    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'expiry': self.expiry if isinstance(self.expiry, str) else self.expiry.isoformat(),
            'strike': self.strike,
            'option_type': self.option_type,
            'quantity': self.quantity
        }


class Portfolio:
    """
    Manages a portfolio of options positions.
    
    Tracks positions, calculates aggregate Greeks, and provides mark-to-market.
    """
    
    def __init__(self, positions: List[Position], initial_capital: float = 10000):
        """
        Initialize portfolio with positions.
        
        Args:
            positions: List of Position objects
            initial_capital: Starting capital (for reference)
        """
        self.positions = positions
        self.initial_capital = initial_capital
    
    def calculate_greeks(self, df: pd.DataFrame, trading_date: date) -> Dict:
        """
        Calculate aggregate portfolio Greeks for a given date.
        
        Uses numpy vectorization for efficiency.
        Portfolio Greeks = Σ(quantity × option_greek × 100)
        
        Args:
            df: Options DataFrame with mid prices
            trading_date: The trading date
            
        Returns:
            Dictionary with aggregated delta, vega, gamma, theta, and portfolio value
        """
        total_delta = 0.0
        total_vega = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_value = 0.0
        
        for pos in self.positions:
            try:
                row = filter_data(df, pos.ticker, trading_date, pos.expiry, pos.strike)
            except ValueError:
                # Position may have expired or data not available
                continue
            
            # Get Greeks and price based on option type
            if pos.option_type == 'call':
                delta = row['call_delta']
                gamma = row['call_gamma']
                vega = row['call_vega']
                theta = row['call_theta']
                price = row['call_mid']
            else:  # put
                delta = row['put_delta']
                gamma = row['put_gamma']
                vega = row['put_vega']
                theta = row['put_theta']
                price = row['put_mid']
            
            # Aggregate: quantity × greek × 100 (contract multiplier)
            multiplier = pos.quantity * 100
            total_delta += delta * multiplier
            total_vega += vega * multiplier
            total_gamma += gamma * multiplier
            total_theta += theta * multiplier
            total_value += price * multiplier
        
        return {
            'delta': total_delta,
            'vega': total_vega,
            'gamma': total_gamma,
            'theta': total_theta,
            'value': total_value
        }
    
    def get_mark_to_market(self, df: pd.DataFrame, trading_date: date) -> float:
        """
        Calculate portfolio mark-to-market value using mid prices.
        
        Args:
            df: Options DataFrame
            trading_date: Trading date
            
        Returns:
            Total portfolio value
        """
        return self.calculate_greeks(df, trading_date)['value']
    
    def get_positions_by_expiry(self) -> Dict[date, List[Position]]:
        """
        Group positions by expiry date.
        
        Returns:
            Dictionary mapping expiry dates to lists of positions
        """
        by_expiry = {}
        for pos in self.positions:
            if pos.expiry not in by_expiry:
                by_expiry[pos.expiry] = []
            by_expiry[pos.expiry].append(pos)
        return by_expiry
    
    def get_unique_expiries(self) -> List[date]:
        """Get sorted list of unique expiry dates in portfolio."""
        return sorted(set(pos.expiry for pos in self.positions))
    
    def get_unique_tickers(self) -> List[str]:
        """Get sorted list of unique tickers in portfolio."""
        return sorted(set(pos.ticker for pos in self.positions))
    
    def to_dict(self) -> Dict:
        """Serialize portfolio to dictionary."""
        return {
            'initial_capital': self.initial_capital,
            'positions': [pos.to_dict() for pos in self.positions]
        }
    
    def __repr__(self) -> str:
        return f"Portfolio({len(self.positions)} positions, capital=${self.initial_capital:,.0f})"


class PortfolioFactory:
    """Factory for creating portfolios from configuration."""
    
    @staticmethod
    def create_atm_straddle(df: pd.DataFrame, ticker: str, expiry: date,
                            initial_capital: float = 10000,
                            quantity_per_leg: Optional[float] = None) -> Portfolio:
        """
        Create a long ATM straddle portfolio sized to initial capital.
        
        Straddle = +1 ATM call + +1 ATM put
        Quantity is auto-calculated from initial_capital if not specified.
        
        Args:
            df: Options DataFrame
            ticker: Ticker symbol
            expiry: Option expiry date
            initial_capital: Starting capital in dollars
            quantity_per_leg: Fixed quantity per leg (overrides auto-sizing)
            
        Returns:
            Portfolio with ATM straddle positions
        """
        # Get first available date for this expiry
        available_dates = get_available_dates(df, ticker, expiry)
        if not available_dates:
            raise ValueError(f"No data available for {ticker} expiry {expiry}")
        
        first_date = available_dates[0]
        atm_strike = get_atm_strike(df, ticker, first_date, expiry)
        row = filter_data(df, ticker, first_date, expiry, atm_strike)
        
        # Calculate straddle cost
        call_mid = row['call_mid']
        put_mid = row['put_mid']
        straddle_cost_per_contract = (call_mid + put_mid) * 100
        
        # Determine quantity
        if quantity_per_leg is not None:
            quantity = quantity_per_leg
        else:
            # Auto-size: how many straddles can we buy with initial capital?
            quantity = initial_capital / straddle_cost_per_contract
        
        # Create positions
        positions = [
            Position(
                ticker=ticker,
                expiry=expiry,
                strike=atm_strike,
                option_type='call',
                quantity=quantity
            ),
            Position(
                ticker=ticker,
                expiry=expiry,
                strike=atm_strike,
                option_type='put',
                quantity=quantity
            )
        ]
        
        return Portfolio(positions, initial_capital)
    
    @staticmethod
    def from_config(config: Dict, df: pd.DataFrame, 
                    ticker: str, expiry: date) -> Portfolio:
        """
        Create portfolio based on configuration.
        
        Args:
            config: Configuration dictionary
            df: Options DataFrame
            ticker: Ticker symbol (used for atm_straddle type)
            expiry: Option expiry (used for atm_straddle type)
            
        Returns:
            Portfolio instance
        """
        portfolio_config = config.get('portfolio', {})
        portfolio_type = portfolio_config.get('type', 'atm_straddle')
        initial_capital = portfolio_config.get('initial_capital', 10000)
        
        if portfolio_type == 'atm_straddle':
            straddle_config = portfolio_config.get('atm_straddle', {})
            quantity_per_leg = straddle_config.get('quantity_per_leg')
            
            return PortfolioFactory.create_atm_straddle(
                df=df,
                ticker=ticker,
                expiry=expiry,
                initial_capital=initial_capital,
                quantity_per_leg=quantity_per_leg
            )
        
        elif portfolio_type == 'custom':
            custom_positions = portfolio_config.get('positions', [])
            
            if not custom_positions:
                raise ValueError("Custom portfolio type requires positions list")
            
            positions = []
            for pos_dict in custom_positions:
                expiry_val = pos_dict['expiry']
                if isinstance(expiry_val, str):
                    expiry_val = pd.to_datetime(expiry_val).date()
                
                positions.append(Position(
                    ticker=pos_dict['ticker'],
                    expiry=expiry_val,
                    strike=pos_dict['strike'],
                    option_type=pos_dict['option_type'],
                    quantity=pos_dict['quantity']
                ))
            
            return Portfolio(positions, initial_capital)
        
        else:
            raise ValueError(f"Unknown portfolio type: {portfolio_type}")
