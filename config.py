"""
Universal configuration for pairs trading strategies
Supports stocks, ETFs, futures, forex, commodities, and crypto pairs
"""
from dataclasses import dataclass, field
from typing import Optional, Literal
import os
from datetime import datetime

@dataclass
class PairConfig:
    """
    Asset pair configuration - the core of what you're trading
    """
    # Asset identifiers
    asset1_ticker: str = 'SPY'  # First asset (e.g., 'SPY', 'GLD', 'CL=F')
    asset2_ticker: str = 'QQQ'  # Second asset (e.g., 'QQQ', 'GDX', 'HO=F')
    pair_name: Optional[str] = None  # Descriptive name (auto-generated if None)
    
    # Asset types (for display/logging)
    asset1_name: str = 'Asset 1'
    asset2_name: str = 'Asset 2'
    
    # Date range
    start_date: str = '2015-01-01'
    end_date: str = datetime.now().strftime('%Y-%m-%d')
    
    # Spread calculation method
    spread_method: Literal['log', 'simple', 'ratio'] = 'log'
    
    def __post_init__(self):
        """Auto-generate pair name if not provided"""
        if self.pair_name is None:
            self.pair_name = f"{self.asset1_ticker}-{self.asset2_ticker}"


@dataclass
class DataConfig:
    """Data quality and preprocessing parameters"""
    # Quality filters
    min_price_threshold: float = 0.1  # Filter out erroneous prices
    outlier_std_threshold: float = 5.0  # Remove extreme outliers (z-score)
    max_missing_days: int = 3  # Max consecutive days to forward-fill
    
    # Statistical validation thresholds
    cointegration_pvalue: float = 0.05  # Engle-Granger test threshold
    stationarity_pvalue: float = 0.05  # ADF test threshold
    min_half_life: float = 5  # Minimum acceptable half-life (days)
    max_half_life: float = 90  # Maximum acceptable half-life (days)
    
    # Data validation
    min_data_points: int = 252  # Minimum trading days required (1 year)
    require_cointegration: bool = True  # Enforce cointegration check


@dataclass
class StrategyConfig:
    """
    Mean-reversion strategy parameters
    Supports multiple entry/exit strategies
    """
    # === Core Parameters ===
    window: int = 30  # Rolling window for z-score calculation
    z_entry_long: float = -2.0  # Enter long when z-score < this
    z_entry_short: float = 2.0  # Enter short when z-score > this
    z_exit: float = 0.5  # Exit when |z-score| < this
    
    # === Advanced Parameters ===
    lookback_period: int = 252  # Long-term statistics (252 = 1 year)
    half_life_max: int = 60  # Max acceptable mean-reversion speed
    
    # === Regime Detection ===
    use_regime_filter: bool = True
    trend_sma_fast: int = 50  # Fast moving average
    trend_sma_slow: int = 200  # Slow moving average
    
    # === Dynamic Thresholds ===
    use_dynamic_thresholds: bool = True
    vol_lookback: int = 100  # Volatility lookback for threshold adjustment
    
    # === Position Management ===
    scale_in_enabled: bool = True
    scale_in_threshold: float = 0.5  # Additional entry if z-score moves further
    max_pyramid_levels: int = 2  # Max number of position additions
    
    # === Hedge Ratio ===
    hedge_ratio_method: Literal['ols', 'tls', 'kalman'] = 'ols'
    rolling_hedge_ratio: bool = False  # Use rolling vs static hedge ratio
    hedge_ratio_window: int = 252  # Window for rolling hedge ratio


@dataclass
class RiskConfig:
    """Comprehensive risk management parameters"""
    # === Capital Allocation ===
    initial_capital: float = 500_000
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.30  # Max 30% of capital per position
    
    # === Stop Loss / Take Profit ===
    use_atr_stops: bool = True
    atr_period: int = 14
    atr_stop_multiple: float = 2.5  # Stop loss at entry ± (2.5 × ATR)
    atr_target_multiple: float = 4.0  # Take profit at entry ± (4.0 × ATR)
    
    # === Portfolio-Level Risk ===
    max_portfolio_heat: float = 0.10  # Max 10% total portfolio risk
    max_correlation_exposure: float = 0.50  # Max 50% in correlated pairs
    max_drawdown_stop: float = 0.20  # Stop trading if DD > 20%
    
    # === Position Limits ===
    max_simultaneous_pairs: int = 5  # Max concurrent pair positions
    min_time_between_trades: int = 1  # Min days between trades (prevent overtrading)


@dataclass
class BacktestConfig:
    """Backtesting and simulation parameters"""
    # === Transaction Costs ===
    transaction_cost_pct: float = 0.0005  # 5 bps per side (0.05%)
    slippage_pct: float = 0.0002  # 2 bps average slippage
    
    # Futures-specific (set to 0 for stocks/ETFs)
    commission_per_contract: float = 0.0  # Set to 2.50 for futures
    contract_multiplier: float = 1.0  # Set to 1000 for CL futures, etc.
    
    # === Backtesting Modes ===
    initial_capital: float = 500_000
    
    # Walk-forward analysis
    walk_forward_enabled: bool = False
    train_window_days: int = 504  # 2 years training
    test_window_days: int = 126  # 6 months testing
    
    # Monte Carlo simulation
    monte_carlo_runs: int = 1000
    monte_carlo_enabled: bool = False
    
    # === Execution Realism ===
    market_impact_enabled: bool = False  # Model market impact for large orders
    use_bid_ask_spread: bool = False  # Use bid/ask instead of mid


@dataclass
class OutputConfig:
    """Output, reporting, and visualization settings"""
    results_dir: str = 'results'
    
    # === Plotting ===
    save_plots: bool = True
    plot_format: str = 'png'  # 'png', 'jpg', 'svg', 'pdf'
    plot_dpi: int = 300
    show_plots: bool = True  # Display plots interactively
    
    # === Reporting ===
    generate_html_report: bool = True
    generate_trade_journal: bool = True
    export_to_csv: bool = True
    
    # === Verbosity ===
    verbose: bool = True
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    
    def __post_init__(self):
        """Create results directory if it doesn't exist"""
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(f"{self.results_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.results_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.results_dir}/data", exist_ok=True)


# ============================================================================
# MASTER CONFIGURATION CLASS
# ============================================================================

class Config:
    """
    Master configuration container for pairs trading system
    
    Usage:
        # Example 1: Default SPY-QQQ
        config = Config()
        
        # Example 2: Custom pair
        config = Config(
            asset1='GLD',
            asset2='GDX',
            pair_name='Gold-Miners'
        )
        
        # Example 3: Oil crack spread (original use case)
        config = Config(
            asset1='CL=F',
            asset2='HO=F',
            pair_name='Oil-Crack-Spread',
            start_date='2010-01-01'
        )
    """
    
    def __init__(self, 
                 asset1: str = 'SPY',
                 asset2: str = 'QQQ',
                 pair_name: Optional[str] = None,
                 start_date: str = '2015-01-01',
                 end_date: Optional[str] = None,
                 initial_capital: float = 500_000):
        """
        Initialize configuration with sensible defaults
        
        Args:
            asset1: First asset ticker
            asset2: Second asset ticker
            pair_name: Optional descriptive name
            start_date: Backtest start date
            end_date: Backtest end date (defaults to today)
            initial_capital: Starting capital for backtest
        """
        # Initialize all config sections
        self.pair = PairConfig(
            asset1_ticker=asset1,
            asset2_ticker=asset2,
            pair_name=pair_name,
            start_date=start_date,
            end_date=end_date or datetime.now().strftime('%Y-%m-%d')
        )
        
        self.data = DataConfig()
        self.strategy = StrategyConfig()
        self.risk = RiskConfig(initial_capital=initial_capital)
        self.backtest = BacktestConfig(initial_capital=initial_capital)
        self.output = OutputConfig()
        
        # Add convenient aliases for backward compatibility
        self.start_date = self.pair.start_date
        self.end_date = self.pair.end_date
        self.initial_capital = initial_capital
        
    def to_dict(self) -> dict:
        """Convert entire config to dictionary"""
        return {
            'pair': self.pair.__dict__,
            'data': self.data.__dict__,
            'strategy': self.strategy.__dict__,
            'risk': self.risk.__dict__,
            'backtest': self.backtest.__dict__,
            'output': self.output.__dict__
        }
    
    def summary(self) -> str:
        """Generate human-readable configuration summary"""
        summary = "\n" + "=" * 70
        summary += f"\n📋 CONFIGURATION SUMMARY: {self.pair.pair_name}"
        summary += "\n" + "=" * 70
        
        summary += "\n\n🎯 TRADING PAIR"
        summary += f"\n   Asset 1: {self.pair.asset1_ticker}"
        summary += f"\n   Asset 2: {self.pair.asset2_ticker}"
        summary += f"\n   Period: {self.pair.start_date} to {self.pair.end_date}"
        summary += f"\n   Spread Method: {self.pair.spread_method}"
        
        summary += "\n\n📊 STRATEGY"
        summary += f"\n   Window: {self.strategy.window} days"
        summary += f"\n   Entry Thresholds: {self.strategy.z_entry_short}σ / {self.strategy.z_entry_long}σ"
        summary += f"\n   Exit Threshold: ±{self.strategy.z_exit}σ"
        summary += f"\n   Regime Filter: {'✅ Enabled' if self.strategy.use_regime_filter else '❌ Disabled'}"
        summary += f"\n   Dynamic Thresholds: {'✅ Enabled' if self.strategy.use_dynamic_thresholds else '❌ Disabled'}"
        summary += f"\n   Position Scaling: {'✅ Enabled' if self.strategy.scale_in_enabled else '❌ Disabled'}"
        
        summary += "\n\n🛡️  RISK MANAGEMENT"
        summary += f"\n   Initial Capital: ${self.risk.initial_capital:,.0f}"
        summary += f"\n   Risk per Trade: {self.risk.risk_per_trade*100:.1f}%"
        summary += f"\n   Max Position: {self.risk.max_position_size*100:.0f}%"
        summary += f"\n   ATR Stops: {'✅ Enabled' if self.risk.use_atr_stops else '❌ Disabled'}"
        if self.risk.use_atr_stops:
            summary += f"\n   Stop: {self.risk.atr_stop_multiple}× ATR | Target: {self.risk.atr_target_multiple}× ATR"
        
        summary += "\n\n💰 TRANSACTION COSTS"
        summary += f"\n   Commission: {self.backtest.transaction_cost_pct*10000:.1f} bps"
        summary += f"\n   Slippage: {self.backtest.slippage_pct*10000:.1f} bps"
        
        summary += "\n\n📈 BACKTESTING"
        summary += f"\n   Walk-Forward: {'✅ Enabled' if self.backtest.walk_forward_enabled else '❌ Disabled'}"
        summary += f"\n   Monte Carlo: {'✅ Enabled' if self.backtest.monte_carlo_enabled else '❌ Disabled'}"
        
        summary += "\n" + "=" * 70 + "\n"
        return summary
    
    def update_from_dict(self, config_dict: dict):
        """Update configuration from dictionary"""
        for section, params in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in params.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    @classmethod
    def from_preset(cls, preset: str) -> 'Config':
        """
        Create configuration from preset templates
        
        Available presets:
            - 'spy-qqq': S&P 500 vs Nasdaq
            - 'oil-crack': CL-HO crude oil crack spread
            - 'gold-silver': Precious metals
            - 'gold-miners': GLD vs GDX
            - 'conservative': Low-risk settings
            - 'aggressive': High-risk settings
        """
        presets = {
            'spy-qqq': {
                'asset1': 'SPY',
                'asset2': 'QQQ',
                'pair_name': 'S&P-Nasdaq',
                'start_date': '2015-01-01'
            },
            'oil-crack': {
                'asset1': 'CL=F',
                'asset2': 'HO=F',
                'pair_name': 'Crude-Heating-Oil',
                'start_date': '2010-01-01'
            },
            'gold-silver': {
                'asset1': 'GLD',
                'asset2': 'SLV',
                'pair_name': 'Gold-Silver',
                'start_date': '2015-01-01'
            },
            'gold-miners': {
                'asset1': 'GLD',
                'asset2': 'GDX',
                'pair_name': 'Gold-Miners',
                'start_date': '2015-01-01'
            },
            'eafe-usa': {
                'asset1': 'SPY',
                'asset2': 'EFA',
                'pair_name': 'US-International',
                'start_date': '2015-01-01'
            }
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")
        
        return cls(**presets[preset])


# ============================================================================
# PRESET CONFIGURATIONS FOR COMMON PAIRS
# ============================================================================

def get_default_config() -> Config:
    """Default SPY-QQQ configuration"""
    return Config()

def get_oil_crack_config() -> Config:
    """Original CL-HO crack spread configuration"""
    config = Config(
        asset1='CL=F',
        asset2='HO=F',
        pair_name='Oil-Crack-Spread',
        start_date='2010-01-01'
    )
    # Futures-specific settings
    config.backtest.commission_per_contract = 2.50
    config.backtest.contract_multiplier = 1000
    return config

def get_gold_miners_config() -> Config:
    """Gold vs Gold Miners ETF"""
    return Config(
        asset1='GLD',
        asset2='GDX',
        pair_name='Gold-Miners',
        start_date='2015-01-01'
    )

def get_conservative_config() -> Config:
    """Conservative risk settings"""
    config = Config()
    config.risk.risk_per_trade = 0.01  # 1% per trade
    config.risk.max_position_size = 0.20  # Max 20%
    config.strategy.z_entry_long = -2.5  # Wider thresholds
    config.strategy.z_entry_short = 2.5
    config.strategy.scale_in_enabled = False  # No pyramiding
    return config

def get_aggressive_config() -> Config:
    """Aggressive risk settings"""
    config = Config()
    config.risk.risk_per_trade = 0.03  # 3% per trade
    config.risk.max_position_size = 0.40  # Max 40%
    config.strategy.z_entry_long = -1.5  # Tighter thresholds
    config.strategy.z_entry_short = 1.5
    config.strategy.scale_in_enabled = True
    config.strategy.max_pyramid_levels = 3
    return config


# ============================================================================
# USAGE EXAMPLES (for testing/documentation)
# ============================================================================

if __name__ == "__main__":
    # Example 1: Default configuration
    config1 = Config()
    print(config1.summary())
    
    # Example 2: Custom pair
    config2 = Config(asset1='GLD', asset2='SLV', pair_name='Gold-Silver')
    print(config2.summary())
    
    # Example 3: From preset
    config3 = Config.from_preset('oil-crack')
    print(config3.summary())
    
    # Example 4: Conservative settings
    config4 = get_conservative_config()
    print(config4.summary())