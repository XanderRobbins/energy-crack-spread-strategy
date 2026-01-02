"""
Configuration module for CL-HO crack spread trading strategy
"""
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class DataConfig:
    """Data fetching and preprocessing configuration"""
    start_date: str = '2015-01-01'
    end_date: str = '2025-01-01'
    cl_ticker: str = 'CL=F'  # WTI Crude Oil Futures
    ho_ticker: str = 'HO=F'  # Heating Oil Futures
    
    # Data quality
    min_price_threshold: float = 0.1  # Filter out erroneous prices
    outlier_std_threshold: float = 5.0  # Remove extreme outliers
    
    # Statistical validation thresholds
    cointegration_pvalue: float = 0.05  # Threshold for cointegration test

@dataclass
class StrategyConfig:
    """Mean-reversion strategy parameters"""
    # Core parameters
    window: int = 30  # Rolling window for mean/std calculation
    z_entry_long: float = -2.0  # Enter long when z-score below this
    z_entry_short: float = 2.0  # Enter short when z-score above this
    z_exit: float = 0.5  # Exit when z-score crosses this threshold
    
    # Enhanced parameters
    lookback_period: int = 252  # Days for long-term mean estimation
    half_life_max: int = 60  # Max acceptable mean-reversion half-life
    
    # Regime detection
    use_regime_filter: bool = True
    trend_sma_fast: int = 50
    trend_sma_slow: int = 200
    
    # Volatility adjustment
    use_dynamic_thresholds: bool = True
    vol_lookback: int = 100
    
    # Position pyramiding
    scale_in_enabled: bool = True
    scale_in_threshold: float = 0.5  # Add to position if z-score increases by this
    max_pyramid_levels: int = 2

@dataclass
class RiskConfig:
    """Risk management parameters"""
    initial_capital: float = 500_000
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.30  # Max 30% of capital per position
    
    # Stop loss / take profit
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_stop_multiple: float = 2.5
    atr_target_multiple: float = 4.0

@dataclass
class BacktestConfig:
    """Backtesting parameters"""
    transaction_cost_pct: float = 0.0005  # 5 bps per side
    slippage_pct: float = 0.0002  # 2 bps slippage
    
    # Commission structure (futures)
    commission_per_contract: float = 2.50
    
    # Walk-forward analysis
    train_test_split: float = 0.7
    rolling_window_days: int = 504  # 2 years
    
    # Initial capital (for backtester)
    initial_capital: float = 500_000

@dataclass
class OutputConfig:
    """Output and reporting configuration"""
    results_dir: str = 'results'
    save_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300
    generate_html_report: bool = True
    
    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)

# Global config object
class Config:
    """Master configuration container"""
    def __init__(self):
        self.data = DataConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.output = OutputConfig()
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'data': self.data.__dict__,
            'strategy': self.strategy.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
            'output': self.output.__dict__
        }