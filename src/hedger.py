"""
Hedging logic for delta and delta-vega hedging.

Uses numpy matrix operations to solve the system of equations for
simultaneous delta-vega neutralization.

Delta Hedge:
    shares = -portfolio_delta

Delta-Vega Hedge (2x2 system):
    | hedge_delta  1 |   | options_qty |   | -portfolio_delta |
    | hedge_vega   0 | × | shares_qty  | = | -portfolio_vega  |
    
    Solved via np.linalg.solve()
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class DeltaHedgeResult:
    """Result of a delta hedge calculation."""
    shares: float              # Target share position (fractional allowed)
    shares_traded: float       # Absolute change from previous position
    notional_traded: float     # Dollar value of shares traded
    transaction_cost: float    # Cost incurred (if with_costs=True)
    
    def to_dict(self) -> Dict:
        return {
            'shares': self.shares,
            'shares_traded': self.shares_traded,
            'notional_traded': self.notional_traded,
            'transaction_cost': self.transaction_cost
        }


@dataclass
class DeltaVegaHedgeResult:
    """Result of a delta-vega hedge calculation."""
    options: float             # Target option position (fractional allowed)
    shares: float              # Target share position
    options_traded: float      # Absolute change in options from previous
    shares_traded: float       # Absolute change in shares from previous
    option_notional_traded: float   # Dollar value of options traded
    share_notional_traded: float    # Dollar value of shares traded
    total_notional_traded: float    # Combined notional
    transaction_cost: float    # Cost incurred (if with_costs=True)
    hedge_option_expiry: Optional[str] = None  # Expiry of hedge option used
    hedge_option_strike: Optional[float] = None  # Strike of hedge option
    
    def to_dict(self) -> Dict:
        return {
            'options': self.options,
            'shares': self.shares,
            'options_traded': self.options_traded,
            'shares_traded': self.shares_traded,
            'option_notional_traded': self.option_notional_traded,
            'share_notional_traded': self.share_notional_traded,
            'total_notional_traded': self.total_notional_traded,
            'transaction_cost': self.transaction_cost,
            'hedge_option_expiry': self.hedge_option_expiry,
            'hedge_option_strike': self.hedge_option_strike
        }


class Hedger:
    """
    Calculates hedge quantities for delta and delta-vega hedging.
    
    Uses numpy linear algebra for the delta-vega system.
    Transaction costs are applied as a percentage of notional traded.
    """
    
    def __init__(self, transaction_cost_pct: float = 5.0):
        """
        Initialize hedger.
        
        Args:
            transaction_cost_pct: Transaction cost as percentage (e.g., 5.0 = 5%)
        """
        self.transaction_cost_pct = transaction_cost_pct
    
    def calculate_transaction_cost(self, notional: float) -> float:
        """
        Calculate transaction cost for a given notional.
        
        Args:
            notional: Dollar value of the trade
            
        Returns:
            Transaction cost in dollars
        """
        return abs(notional) * (self.transaction_cost_pct / 100)
    
    def hedge_delta(self, portfolio_delta: float, underlying_price: float,
                    prev_shares: float = 0.0, 
                    with_costs: bool = False) -> DeltaHedgeResult:
        """
        Calculate delta hedge using underlying stock.
        
        Delta hedge: shares = -portfolio_delta
        (If portfolio has +100 delta, short 100 shares to neutralize)
        
        Args:
            portfolio_delta: Current portfolio delta exposure
            underlying_price: Current stock price
            prev_shares: Previous share position (for calculating trades)
            with_costs: Whether to apply transaction costs
            
        Returns:
            DeltaHedgeResult with hedge details
        """
        # Target share position to neutralize delta
        target_shares = -portfolio_delta
        
        # Shares to trade (change from previous position)
        shares_traded = abs(target_shares - prev_shares)
        
        # Notional value of trade
        notional_traded = shares_traded * underlying_price
        
        # Transaction cost
        if with_costs:
            transaction_cost = self.calculate_transaction_cost(notional_traded)
        else:
            transaction_cost = 0.0
        
        return DeltaHedgeResult(
            shares=target_shares,
            shares_traded=shares_traded,
            notional_traded=notional_traded,
            transaction_cost=transaction_cost
        )
    
    def hedge_delta_vega(self, portfolio_delta: float, portfolio_vega: float,
                         hedge_option: Dict, underlying_price: float,
                         prev_shares: float = 0.0, prev_options: float = 0.0,
                         with_costs: bool = False) -> DeltaVegaHedgeResult:
        """
        Calculate delta-vega hedge using options and stock.
        
        Solves the 2x2 system:
            A @ x = b
        where:
            A = [[hedge_delta, 1], [hedge_vega, 0]]
            x = [options_qty, shares_qty]
            b = [-portfolio_delta, -portfolio_vega]
        
        Args:
            portfolio_delta: Current portfolio delta exposure
            portfolio_vega: Current portfolio vega exposure
            hedge_option: Dict with hedge option Greeks (delta, vega, mid_price, etc.)
            underlying_price: Current stock price
            prev_shares: Previous share position
            prev_options: Previous option position
            with_costs: Whether to apply transaction costs
            
        Returns:
            DeltaVegaHedgeResult with hedge details
        """
        hedge_delta = hedge_option['delta']
        hedge_vega = hedge_option['vega']
        hedge_price = hedge_option['mid_price']
        
        # Check for degenerate case (zero vega in hedge option)
        if abs(hedge_vega) < 1e-10:
            # Fall back to delta-only hedge
            delta_result = self.hedge_delta(
                portfolio_delta, underlying_price, prev_shares, with_costs
            )
            return DeltaVegaHedgeResult(
                options=0.0,
                shares=delta_result.shares,
                options_traded=abs(prev_options),
                shares_traded=delta_result.shares_traded,
                option_notional_traded=abs(prev_options) * hedge_price * 100,
                share_notional_traded=delta_result.notional_traded,
                total_notional_traded=delta_result.notional_traded + abs(prev_options) * hedge_price * 100,
                transaction_cost=delta_result.transaction_cost,
                hedge_option_expiry=str(hedge_option.get('expiry', '')),
                hedge_option_strike=hedge_option.get('strike')
            )
        
        # Build the system: A @ x = b
        # Row 1: hedge_delta * options + 1 * shares = -portfolio_delta
        # Row 2: hedge_vega * options + 0 * shares = -portfolio_vega
        A = np.array([
            [hedge_delta * 100, 1.0],    # *100 for contract multiplier on delta
            [hedge_vega * 100, 0.0]      # *100 for contract multiplier on vega
        ])
        b = np.array([-portfolio_delta, -portfolio_vega])
        
        # Solve the system
        try:
            x = np.linalg.solve(A, b)
            target_options = x[0]  # Number of option contracts
            target_shares = x[1]   # Number of shares
        except np.linalg.LinAlgError:
            # Singular matrix - fall back to delta-only
            target_options = 0.0
            target_shares = -portfolio_delta
        
        # Calculate trades (changes from previous positions)
        options_traded = abs(target_options - prev_options)
        shares_traded = abs(target_shares - prev_shares)
        
        # Calculate notional values
        option_notional = options_traded * hedge_price * 100  # Per contract
        share_notional = shares_traded * underlying_price
        total_notional = option_notional + share_notional
        
        # Transaction cost
        if with_costs:
            transaction_cost = self.calculate_transaction_cost(total_notional)
        else:
            transaction_cost = 0.0
        
        return DeltaVegaHedgeResult(
            options=target_options,
            shares=target_shares,
            options_traded=options_traded,
            shares_traded=shares_traded,
            option_notional_traded=option_notional,
            share_notional_traded=share_notional,
            total_notional_traded=total_notional,
            transaction_cost=transaction_cost,
            hedge_option_expiry=str(hedge_option.get('expiry', '')),
            hedge_option_strike=hedge_option.get('strike')
        )
    
    def calculate_hedge_pnl(self, prev_shares: float, prev_options: float,
                            prev_underlying_price: float, prev_option_price: float,
                            curr_underlying_price: float, curr_option_price: float) -> Dict:
        """
        Calculate P&L from hedge positions.
        
        Args:
            prev_shares: Share position at start of period
            prev_options: Option position at start of period
            prev_underlying_price: Stock price at start
            prev_option_price: Option mid price at start
            curr_underlying_price: Stock price at end
            curr_option_price: Option mid price at end
            
        Returns:
            Dictionary with share_pnl, option_pnl, total_pnl
        """
        # Share P&L: position * price change
        share_pnl = prev_shares * (curr_underlying_price - prev_underlying_price)
        
        # Option P&L: position * price change * 100 (contract multiplier)
        option_pnl = prev_options * (curr_option_price - prev_option_price) * 100
        
        return {
            'share_pnl': share_pnl,
            'option_pnl': option_pnl,
            'total_pnl': share_pnl + option_pnl
        }
