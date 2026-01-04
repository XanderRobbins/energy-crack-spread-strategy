# Universal Pairs Trading System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

**A professional-grade quantitative trading system for statistical arbitrage across any cointegrated asset pair**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Results](#-sample-results)

</div>

---

## Overview

The **Universal Pairs Trading System** is a sophisticated mean-reversion strategy framework designed for cointegrated asset pairs. Built with institutional-grade risk management and comprehensive statistical validation, it works seamlessly across stocks, ETFs, futures, commodities, and cryptocurrencies.

### What is Pairs Trading?

Pairs trading is a market-neutral statistical arbitrage strategy that exploits temporary price divergences between two historically correlated assets. When the spread deviates from its historical mean, the strategy goes long the undervalued asset and short the overvalued one, profiting when the spread reverts.

---

## Features

### Statistical Validation Suite
- **Cointegration Testing** - Engle-Granger methodology
- **Stationarity Analysis** - Augmented Dickey-Fuller test
- **Half-Life Calculation** - Mean-reversion speed estimation
- **Rolling Cointegration** - Dynamic regime detection

### Advanced Strategy Components
- **Dynamic Z-Score Thresholds** - Volatility-adjusted entry/exit
- **Regime Detection** - Distinguish mean-reverting vs. trending markets
- **Momentum Filters** - Avoid catching falling knives
- **Position Pyramiding** - Scale into high-confidence trades
- **Multiple Exit Strategies** - Mean-reversion, stop-loss, and regime-based

### Risk Management
- **ATR-Based Position Sizing** - Volatility-adjusted exposure
- **Dynamic Stop-Loss/Take-Profit** - Adaptive risk parameters
- **Portfolio Heat Management** - Maximum drawdown protection
- **Circuit Breakers** - Automatic trading halt on catastrophic losses
- **Kelly Criterion** - Optimal position sizing (optional)

### Backtesting & Analysis
- **Realistic Transaction Costs** - Commission, slippage, and market impact
- **Walk-Forward Optimization** - Out-of-sample validation
- **Monte Carlo Simulation** - Risk assessment via bootstrapping
- **Comprehensive Metrics** - Sharpe, Sortino, Calmar, drawdown, win rate
- **Trade Journal Export** - Detailed CSV for further analysis

### Visualization
- Publication-quality charts with institutional aesthetics
- Equity curves with drawdown
- Trade distribution analysis
- Monthly returns heatmap
- Rolling performance metrics
- Regime classification plots

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/universal-pairs-trading.git
cd universal-pairs-trading

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
yfinance>=0.2.0
statsmodels>=0.14.0
scipy>=1.10.0
```

---

## Quick Start

### Interactive Mode

```bash
python main.py
```

Select from pre-configured examples:
- SPY-QQQ - S&P 500 vs Nasdaq ETFs
- GLD-SLV - Gold vs Silver
- CL-HO - Crude Oil crack spread (futures)
- AAPL-MSFT - Tech giants
- Multi-Pair Comparison - Portfolio analysis
- Custom Pair - Enter your own

### Programmatic Usage

```python
from main import run_pairs_strategy

# Example 1: Basic usage
result = run_pairs_strategy(
    asset1='SPY',
    asset2='QQQ',
    pair_name='SPY-QQQ',
    start_date='2015-01-01',
    initial_capital=500_000
)

# Example 2: Your original oil crack spread
result = run_pairs_strategy(
    asset1='CL=F',
    asset2='HO=F',
    pair_name='Oil-Crack-Spread',
    start_date='2010-01-01',
    asset_type='futures'
)

# Example 3: Multi-pair portfolio
from main import run_multiple_pairs

pairs = [
    ('SPY', 'QQQ', 'S&P-Nasdaq'),
    ('GLD', 'SLV', 'Gold-Silver'),
    ('GLD', 'GDX', 'Gold-Miners')
]

comparison = run_multiple_pairs(pairs, start_date='2015-01-01')
```

---

## Documentation

### Project Structure

```
universal-pairs-trading/
├── config.py              # Configuration management
├── data_handler.py        # Data acquisition & validation
├── strategy.py            # Signal generation logic
├── backtester.py          # Backtesting engine
├── risk_manager.py        # Risk management system
├── visualization.py       # Charting and reporting
├── main.py               # Orchestration script
├── requirements.txt      # Dependencies
└── results/              # Output directory (auto-created)
```

### Core Components

#### 1. Configuration (config.py)

```python
from config import Config

# Default configuration
config = Config(asset1='SPY', asset2='QQQ')

# Custom configuration
config = Config(
    asset1='GLD',
    asset2='SLV',
    start_date='2015-01-01',
    initial_capital=1_000_000
)

# From preset
config = Config.from_preset('oil-crack')
```

#### 2. Data Handler (data_handler.py)

```python
from data_handler import PairsDataHandler

handler = PairsDataHandler(config, 'SPY', 'QQQ')

# Fetch and validate
df = handler.fetch_data()
spread = handler.compute_spread(method='log')

# Statistical tests
handler.test_cointegration()
handler.test_stationarity(spread)
handler.calculate_half_life(spread)
handler.calculate_hedge_ratio()

# Rolling analysis
rolling_coint = handler.calculate_rolling_cointegration(window=252)
```

#### 3. Strategy (strategy.py)

```python
from strategy import PairsMeanReversionStrategy

strategy = PairsMeanReversionStrategy(config)
signals = strategy.generate_signals(df, spread, rolling_coint)

# Analyze trades
trades = strategy.get_trade_list()
analysis = strategy.analyze_trades()
```

#### 4. Backtester (backtester.py)

```python
from backtester import PairsBacktester

backtester = PairsBacktester(config)
results = backtester.run_backtest(signals, initial_capital=500_000)
metrics = backtester.calculate_performance_metrics()

# Advanced analysis
mc_results = backtester.monte_carlo_simulation(n_simulations=1000)
benchmark = backtester.benchmark_comparison(benchmark_ticker='SPY')
```

#### 5. Risk Management (risk_manager.py)

```python
from risk_manager import PairsRiskManager

risk_mgr = PairsRiskManager(config)
risk_adjusted = risk_mgr.apply_risk_filters(df, signals, asset_type='etf')

# Risk metrics
var_95 = risk_mgr.calculate_var(returns, confidence=0.95)
cvar_95 = risk_mgr.calculate_cvar(returns, confidence=0.95)
kelly_pct = risk_mgr.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
```

#### 6. Visualization (visualization.py)

```python
from visualization import PairsVisualizer

viz = PairsVisualizer(config, pair_name='SPY-QQQ')

# Generate plots
viz.plot_price_series(df, save_path='results/prices.png')
viz.plot_spread_analysis(signals, save_path='results/spread.png')
viz.plot_equity_curve(results, save_path='results/equity.png')
viz.plot_trade_distribution(trades, save_path='results/trades.png')
viz.plot_monthly_returns_heatmap(results, save_path='results/monthly.png')
```
---

## Methodology

### Entry Logic

1. Calculate z-score of spread: `z = (spread - mean) / std`
2. Adjust thresholds dynamically based on volatility regime
3. Confirm favorable market regime (mean-reverting vs. trending)
4. Verify momentum indicators align with entry direction
5. Optional: Only trade during cointegrated periods

**Entry Signals:**
- **Long:** z-score < -2.0σ (spread oversold)
- **Short:** z-score > +2.0σ (spread overbought)

### Exit Logic

- **Mean Reversion:** |z-score| < 0.5σ (primary exit)
- **Stop Loss:** z-score reverses direction sharply
- **Regime Change:** Market shifts to volatile/trending
- **Time-Based:** Maximum holding period exceeded
- **ATR Stops:** Dynamic stop-loss based on volatility

### Position Sizing

Using ATR-based risk management:

```
Position Size = (Capital × Risk%) / (ATR × Stop Multiple)
```

**Constraints:**
- Max 30% of capital per position
- Max 10% total portfolio heat
- Volatility-adjusted scaling during extreme regimes

---

## Statistical Foundation

### Why Cointegration Matters

Two price series can be correlated but still drift apart permanently. Cointegration ensures:
- A stable long-term equilibrium relationship exists
- Deviations are temporary and statistically predictable
- Mean-reversion is not just correlation noise

### Validation Metrics

- **P-Value < 0.05:** Statistically significant cointegration
- **ADF Statistic < Critical Value:** Spread is stationary
- **Half-Life < 60 days:** Fast enough for practical trading
- **Hedge Ratio (β):** Optimal units of Asset2 per unit of Asset1

---

## Configuration Options

### Strategy Parameters

```python
config.strategy.window = 30                    # Z-score calculation window
config.strategy.z_entry_long = -2.0           # Long entry threshold
config.strategy.z_entry_short = 2.0           # Short entry threshold
config.strategy.z_exit = 0.5                  # Exit threshold
config.strategy.use_regime_filter = True      # Enable regime detection
config.strategy.use_dynamic_thresholds = True # Volatility-adjusted
config.strategy.scale_in_enabled = True       # Position pyramiding
```

### Risk Management

```python
config.risk.risk_per_trade = 0.02            # 2% risk per trade
config.risk.max_position_size = 0.30         # Max 30% per position
config.risk.atr_period = 14                  # ATR calculation period
config.risk.atr_stop_multiple = 2.5          # Stop at 2.5× ATR
config.risk.atr_target_multiple = 4.0        # Target at 4.0× ATR
```

### Transaction Costs

```python
config.backtest.transaction_cost_pct = 0.0005  # 5 bps per side
config.backtest.slippage_pct = 0.0002         # 2 bps slippage
config.backtest.commission_per_contract = 2.50 # For futures
```

---

## Validation & Testing

### Walk-Forward Analysis

```python
backtester.walk_forward_analysis(
    df=df,
    strategy_func=generate_signals,
    train_window=504,  # 2 years
    test_window=126    # 6 months
)
```

### Monte Carlo Simulation

```python
mc_results = backtester.monte_carlo_simulation(
    n_simulations=1000,
    n_trades=100
)

print(f"Probability of Profit: {mc_results['summary']['Probability_of_Profit']:.1f}%")
print(f"5th Percentile Return: {mc_results['summary']['Percentile_5_Return']:.2f}%")
print(f"95th Percentile Return: {mc_results['summary']['Percentile_95_Return']:.2f}%")
```

---

## Use Cases

### 1. Equity Pairs
- SPY vs QQQ (S&P 500 vs Nasdaq)
- EFA vs EEM (Developed vs Emerging Markets)
- XLE vs XLF (Energy vs Financials)

### 2. Precious Metals
- GLD vs SLV (Gold vs Silver)
- GLD vs GDX (Gold vs Gold Miners)

### 3. Energy / Commodities
- CL vs HO (Crude Oil crack spread)
- CL vs NG (Crude vs Natural Gas)
- USO vs UNG (Oil vs Gas ETFs)

### 4. Cryptocurrencies
- BTC-USD vs ETH-USD
- BTC-USD vs BNB-USD

### 5. Sector Rotations
- Tech vs Healthcare
- Utilities vs Industrials

---

## Risk Disclosure

**This software is for educational and research purposes only.**

- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Cointegration relationships can break down unexpectedly
- Always backtest thoroughly on out-of-sample data
- Use appropriate position sizing and risk management
- Consult a licensed financial advisor before trading

---

## Advanced Features

### Custom Strategy Development

Extend the `PairsMeanReversionStrategy` class:

```python
class MyCustomStrategy(PairsMeanReversionStrategy):
    def _generate_entry_signals(self, df, rolling_coint=None):
        # Your custom entry logic
        df['Signal'] = 0
        
        # Example: Add RSI filter
        df['RSI'] = self.calculate_rsi(df['Spread'], period=14)
        
        long_condition = (
            (df['Z_Score'] < -2.0) &
            (df['RSI'] < 30) &  # Oversold
            (df['Regime'] == 'Mean_Reverting')
        )
        
        df.loc[long_condition, 'Signal'] = 1
        return df
```

### Integration with Live Trading

```python
# Pseudo-code for live trading integration
from live_trading import BrokerAPI

broker = BrokerAPI(api_key='your_key')

# Get current positions
positions = broker.get_positions()

# Generate signals
signals = strategy.generate_signals(df_latest, spread_latest)

# Execute trades
if signals['Signal'].iloc[-1] == 1:
    broker.submit_order(
        symbol='SPY',
        qty=calculated_position_size,
        side='buy'
    )
    broker.submit_order(
        symbol='QQQ',
        qty=calculated_position_size * hedge_ratio,
        side='sell'
    )
```

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/universal-pairs-trading.git
cd universal-pairs-trading

# Create feature branch
git checkout -b feature/amazing-feature

# Make changes and test
python -m pytest tests/

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{robbins2026pairs,
  author = {Robbins, Alexander},
  title = {Universal Pairs Trading System: A Professional-Grade Statistical Arbitrage Framework},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/XanderRobbins/universal-pairs-trading}
}
```

---

## Author

**Alexander Robbins**  
University of Florida | Math, CS, Economics  
📧 robbins.a@ufl.edu  
🔗 [GitHub](https://github.com/XanderRobbins) | [LinkedIn](https://www.linkedin.com/in/alexander-robbins-a1086a248/) | [Website](https://xanderrobbins.github.io/)

---


## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- University of Florida Department of Mathematics
- Open-source Python community (pandas, numpy, matplotlib, statsmodels)
