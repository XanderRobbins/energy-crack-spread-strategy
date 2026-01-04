"""
Comprehensive risk management system for pairs trading strategies
Implements ATR-based position sizing, stop-loss, and portfolio controls
Works with any asset pair: stocks, ETFs, futures, commodities, crypto
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class PairsRiskManager:
    """
    Advanced risk management with multiple safety layers
    
    Features:
    - ATR-based position sizing
    - Dynamic stop-loss and take-profit levels
    - Portfolio heat management
    - Maximum position limits
    - Correlation-based exposure control
    - Drawdown protection
    - Works with ANY asset pair
    """
    
    def __init__(self, config, pair_name: Optional[str] = None):
        """
        Initialize risk manager
        
        Args:
            config: Configuration object
            pair_name: Optional descriptive name for the pair
        """
        self.config = config
        self.pair_name = pair_name or config.pair.pair_name
        self.current_positions = {}
        self.equity_curve = []
        self.peak_equity = config.risk.initial_capital
        
    def calculate_atr(self, high: pd.Series, low: pd.Series, 
                     close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (Wilder's method)
        
        ATR measures market volatility by decomposing the entire range of an asset
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Lookback period for ATR calculation
        
        Returns:
            pd.Series: ATR values
        """
        # True Range components
        tr1 = high - low  # High - Low
        tr2 = abs(high - close.shift(1))  # |High - Previous Close|
        tr3 = abs(low - close.shift(1))   # |Low - Previous Close|
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the exponential moving average of TR (Wilder's smoothing)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def calculate_position_size(self, df: pd.DataFrame, 
                               signal: pd.Series,
                               asset_type: str = 'stock') -> pd.Series:
        """
        Calculate optimal position size with clear hierarchy of constraints
        
        Args:
            df: DataFrame with market data (Asset1/Asset2)
            signal: Trading signals
            asset_type: 'stock', 'etf', 'futures', 'crypto'
        
        Returns:
            pd.Series: Position sizes
        """
        # Calculate ATR for volatility adjustment
        atr_asset1 = self.calculate_atr(
            df['Asset1_High'], df['Asset1_Low'], df['Asset1_Close'],
            period=self.config.risk.atr_period
        )
        
        atr_asset2 = self.calculate_atr(
            df['Asset2_High'], df['Asset2_Low'], df['Asset2_Close'],
            period=self.config.risk.atr_period
        )
        
        avg_atr = (atr_asset1 + atr_asset2) / 2
        
        # Step 1: Calculate base position size from risk tolerance
        risk_amount = self.config.risk.initial_capital * self.config.risk.risk_per_trade
        position_value = risk_amount / (avg_atr * self.config.risk.atr_stop_multiple)
        
        # Asset-specific multiplier (futures use contract size)
        if asset_type == 'futures':
            contract_size = self.config.backtest.contract_multiplier
            position_size = position_value / (df['Asset1_Close'] * contract_size)
        else:
            # For stocks/ETFs: calculate number of shares/units
            position_size = position_value / df['Asset1_Close']
        
        # Step 2: Apply hard limits (in order of priority)
        
        # 2a. Maximum contracts/shares limit
        if asset_type == 'futures':
            MAX_CONTRACTS = 10
            position_size = np.minimum(position_size, MAX_CONTRACTS)
        else:
            MAX_SHARES = 10000
            position_size = np.minimum(position_size, MAX_SHARES)
        
        # 2b. Capital percentage limit
        MAX_CAPITAL_PCT = 0.20
        if asset_type == 'futures':
            capital_limit = (
                self.config.risk.initial_capital * MAX_CAPITAL_PCT
            ) / (df['Asset1_Close'] * self.config.backtest.contract_multiplier)
        else:
            capital_limit = (
                self.config.risk.initial_capital * MAX_CAPITAL_PCT
            ) / df['Asset1_Close']
        position_size = np.minimum(position_size, capital_limit)
        
        # 2c. Portfolio-level maximum position from config
        if asset_type == 'futures':
            config_max = (
                self.config.risk.initial_capital * self.config.risk.max_position_size
            ) / (df['Asset1_Close'] * self.config.backtest.contract_multiplier)
        else:
            config_max = (
                self.config.risk.initial_capital * self.config.risk.max_position_size
            ) / df['Asset1_Close']
        position_size = np.minimum(position_size, config_max)
        
        # Step 3: Round and ensure minimum size
        if asset_type == 'futures':
            min_size = 1  # Minimum 1 contract
        else:
            min_size = 1  # Minimum 1 share
            
        position_size = np.where(
            abs(signal) > 0,
            np.maximum(np.round(position_size), min_size),
            0
        )
        
        # Step 4: Apply signal direction
        position_size = position_size * np.sign(signal)
        
        return pd.Series(position_size, index=df.index)
    
    def set_stop_loss_take_profit(self, df: pd.DataFrame, 
                                   entry_price: float,
                                   direction: int,
                                   use_asset1: bool = True) -> Tuple[float, float]:
        """
        Calculate dynamic stop-loss and take-profit levels
        
        Args:
            df: DataFrame with current market data
            entry_price: Entry price for the position
            direction: 1 for long, -1 for short
            use_asset1: Use Asset1 for ATR (default True)
        
        Returns:
            Tuple of (stop_loss, take_profit) prices
        """
        # Get current ATR (use Asset1 by default)
        if use_asset1:
            current_atr = self.calculate_atr(
                df['Asset1_High'], 
                df['Asset1_Low'], 
                df['Asset1_Close'],
                period=self.config.risk.atr_period
            ).iloc[-1]
        else:
            current_atr = self.calculate_atr(
                df['Asset2_High'], 
                df['Asset2_Low'], 
                df['Asset2_Close'],
                period=self.config.risk.atr_period
            ).iloc[-1]
        
        if direction == 1:  # Long position
            stop_loss = entry_price - (current_atr * self.config.risk.atr_stop_multiple)
            take_profit = entry_price + (current_atr * self.config.risk.atr_target_multiple)
        else:  # Short position
            stop_loss = entry_price + (current_atr * self.config.risk.atr_stop_multiple)
            take_profit = entry_price - (current_atr * self.config.risk.atr_target_multiple)
        
        return stop_loss, take_profit
    
    def calculate_portfolio_heat(self, open_positions: Dict) -> float:
        """
        Calculate total portfolio heat (sum of all risk across open positions)
        
        Args:
            open_positions: Dictionary of open positions
        
        Returns:
            float: Portfolio heat as percentage of capital
        """
        total_heat = 0
        
        for position_id, position_data in open_positions.items():
            # Heat = (Entry Price - Stop Loss) * Position Size / Capital
            entry = position_data['entry_price']
            stop = position_data['stop_loss']
            size = position_data['size']
            
            risk_per_contract = abs(entry - stop) * size
            total_heat += risk_per_contract
        
        return total_heat / self.config.risk.initial_capital
    
    def check_drawdown_limit(self, current_equity: float) -> bool:
        """
        Check if maximum drawdown limit has been breached
        
        Args:
            current_equity: Current portfolio value
        
        Returns:
            bool: True if trading should stop due to drawdown
        """
        self.peak_equity = max(self.peak_equity, current_equity)
        current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
        
        # Get max allowed drawdown from config
        max_allowed_drawdown = self.config.risk.max_drawdown_stop
        
        return current_drawdown > max_allowed_drawdown
    
        
    def apply_risk_filters(self, df: pd.DataFrame, 
                        signals: pd.DataFrame,
                        asset_type: str = 'stock') -> pd.DataFrame:
        """Apply comprehensive risk filters to trading signals"""
        print("\n" + "=" * 60)
        print(f"🛡️  APPLYING RISK MANAGEMENT: {self.pair_name}")
        print("=" * 60)
        
        # Create a copy to avoid modifying original
        risk_adjusted = signals.copy()
        
        # ✅ FIX: Align df with signals index to prevent shape mismatch
        df_aligned = df.loc[signals.index]  # ← ADD THIS LINE
        
        # 1. Calculate position sizes
        print("\n1️⃣  Calculating position sizes...")
        risk_adjusted['Position_Size'] = self.calculate_position_size(
            df_aligned,  # ← USE df_aligned instead of df
            risk_adjusted['Position'],
            asset_type=asset_type
        )
        
        # Rest of the method...
        # Also update other df references to df_aligned:
        
        # 2. Calculate ATR for stop-loss/take-profit
        print("2️⃣  Setting stop-loss and take-profit levels...")
        atr_asset1 = self.calculate_atr(
            df_aligned['Asset1_High'],   # ← df_aligned
            df_aligned['Asset1_Low'],    # ← df_aligned
            df_aligned['Asset1_Close'],  # ← df_aligned
            period=self.config.risk.atr_period
        )
        
    # Continue with df_aligned throughout...

        
        # Calculate stop-loss and take-profit for each row
        risk_adjusted['ATR'] = atr_asset1
        risk_adjusted['Stop_Loss'] = np.where(
            risk_adjusted['Position'] > 0,
            df['Asset1_Close'] - (atr_asset1 * self.config.risk.atr_stop_multiple),
            np.where(
                risk_adjusted['Position'] < 0,
                df['Asset1_Close'] + (atr_asset1 * self.config.risk.atr_stop_multiple),
                np.nan
            )
        )
        
        risk_adjusted['Take_Profit'] = np.where(
            risk_adjusted['Position'] > 0,
            df['Asset1_Close'] + (atr_asset1 * self.config.risk.atr_target_multiple),
            np.where(
                risk_adjusted['Position'] < 0,
                df['Asset1_Close'] - (atr_asset1 * self.config.risk.atr_target_multiple),
                np.nan
            )
        )
        
        # 3. Calculate risk per trade
        if asset_type == 'futures':
            multiplier = self.config.backtest.contract_multiplier
        else:
            multiplier = 1.0
            
        risk_adjusted['Risk_Per_Trade'] = (
            abs(df['Asset1_Close'] - risk_adjusted['Stop_Loss']) * 
            risk_adjusted['Position_Size'] * multiplier
        ) / self.config.risk.initial_capital
        
        # 4. Apply maximum heat filter
        print("3️⃣  Applying portfolio heat limits...")
        risk_adjusted['Cumulative_Heat'] = risk_adjusted['Risk_Per_Trade'].rolling(
            window=20, min_periods=1
        ).sum()
        
        # Reduce position if cumulative heat exceeds threshold
        max_heat = self.config.risk.max_portfolio_heat
        risk_adjusted['Position_Size'] = np.where(
            risk_adjusted['Cumulative_Heat'] > max_heat,
            risk_adjusted['Position_Size'] * 0.5,  # Cut position in half
            risk_adjusted['Position_Size']
        )
        
        # 5. Volatility regime filter
        print("4️⃣  Applying volatility regime filters...")
        risk_adjusted['ATR_Percentile'] = atr_asset1.rolling(
            window=252, min_periods=60
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Reduce position size during extreme volatility (top 10%)
        risk_adjusted['Position_Size'] = np.where(
            risk_adjusted['ATR_Percentile'] > 0.90,
            risk_adjusted['Position_Size'] * 0.5,
            risk_adjusted['Position_Size']
        )
        
        # 6. Add risk metadata
        risk_adjusted['Dollar_Risk'] = (
            abs(df['Asset1_Close'] - risk_adjusted['Stop_Loss']) * 
            risk_adjusted['Position_Size'] * multiplier
        )
        
        risk_adjusted['Reward_Risk_Ratio'] = (
            abs(risk_adjusted['Take_Profit'] - df['Asset1_Close']) /
            abs(df['Asset1_Close'] - risk_adjusted['Stop_Loss'])
        )
        
        self._print_risk_summary(risk_adjusted)
        
        return risk_adjusted
    
    def _print_risk_summary(self, df: pd.DataFrame):
        """Print summary of risk management application"""
        
        active_positions = df[df['Position'] != 0]
        
        if len(active_positions) == 0:
            print("\n⚠️  No active positions after risk filtering")
            return
        
        print("\n" + "=" * 60)
        print(f"📊 RISK MANAGEMENT SUMMARY: {self.pair_name}")
        print("=" * 60)
        print(f"Active positions:           {len(active_positions)}")
        print(f"Average position size:      {active_positions['Position_Size'].mean():.2f} units")
        print(f"Max position size:          {active_positions['Position_Size'].max():.2f} units")
        print(f"Average risk per trade:     {active_positions['Risk_Per_Trade'].mean()*100:.2f}%")
        print(f"Max portfolio heat:         {active_positions['Cumulative_Heat'].max()*100:.2f}%")
        print(f"Average reward/risk ratio:  {active_positions['Reward_Risk_Ratio'].mean():.2f}:1")
        print(f"High volatility periods:    {(active_positions['ATR_Percentile'] > 0.90).sum()} days")
        print("=" * 60)
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)
        
        Returns:
            float: VaR value
        """
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall)
        
        Args:
            returns: Series of returns
            confidence: Confidence level (default 95%)
        
        Returns:
            float: CVaR value
        """
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def calculate_kelly_criterion(self, win_rate: float, 
                                  avg_win: float, 
                                  avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        
        Kelly% = W - [(1-W) / R]
        Where:
            W = Win rate
            R = Average win / Average loss
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
        
        Returns:
            float: Kelly percentage (typically use 1/4 or 1/2 Kelly for safety)
        """
        if avg_loss == 0:
            return 0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Use fractional Kelly (1/4 Kelly) for safety
        fractional_kelly = kelly * 0.25
        
        return max(0, fractional_kelly)  # Never go negative


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from config import Config
    from data_handler import PairsDataHandler
    from strategy import PairsMeanReversionStrategy
    
    # Example: Risk management for SPY-QQQ
    config = Config(asset1='SPY', asset2='QQQ')
    
    # Fetch data
    handler = PairsDataHandler(config, 'SPY', 'QQQ')
    df = handler.fetch_data()
    spread = handler.compute_spread(method='log')
    
    # Generate signals
    strategy = PairsMeanReversionStrategy(config)
    signals = strategy.generate_signals(df, spread)
    
    # Apply risk management
    risk_manager = PairsRiskManager(config, pair_name='SPY-QQQ')
    risk_adjusted_signals = risk_manager.apply_risk_filters(
        df, 
        signals,
        asset_type='etf'  # SPY/QQQ are ETFs
    )
    
    print("\n✅ Risk management applied successfully!")