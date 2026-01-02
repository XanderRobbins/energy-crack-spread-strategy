"""
CL-HO Crack Spread Mean-Reversion Trading Strategy
Main execution script

Author: Alexander Robbins
Date: 2025
"""

from logging import config
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.config import Config
from src.data_handler import DataHandler
from src.strategy import CrackSpreadStrategy
from src.risk_manager import RiskManager
from src.backtester import Backtester
from src.visualization import Visualizer

def main():
    """
    Main execution function for the CL-HO crack spread strategy
    """
    print("\n" + "=" * 70)
    print("🚀 CL-HO CRACK SPREAD TRADING STRATEGY")
    print("=" * 70)
    print("Author: Alexander Robbins")
    print("Strategy: Mean-Reversion on Crude Oil / Heating Oil Spread")
    print("=" * 70)
    
    # === 1. INITIALIZE CONFIGURATION ===
    print("\n📋 Step 1: Loading Configuration...")
    config = Config()
    
    print("\nConfiguration Summary:")
    print(f"  Data Period: {config.data.start_date} to {config.data.end_date}")
    print(f"  Initial Capital: ${config.risk.initial_capital:,.0f}")
    print(f"  Risk per Trade: {config.risk.risk_per_trade * 100}%")
    print(f"  Z-Score Entry Thresholds: [{config.strategy.z_entry_long}, {config.strategy.z_entry_short}]")
    
    # === 2. FETCH AND VALIDATE DATA ===
    print("\n" + "=" * 70)
    data_handler = DataHandler(config.data)
    
    try:
        df = data_handler.fetch_data(verbose=True)
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        return
    
    # Compute crack spread
    spread = data_handler.compute_crack_spread(method='log')
    
    # === 3. STATISTICAL VALIDATION ===
    print("\n" + "=" * 70)
    
    # Test cointegration
    coint_results = data_handler.test_cointegration(verbose=True)
    
    # Test stationarity
    stat_results = data_handler.test_stationarity(spread, verbose=True)
    
    # Calculate half-life
    half_life = data_handler.calculate_half_life(spread, verbose=True)
    
    # Calculate hedge ratio
    hedge_ratio = data_handler.calculate_hedge_ratio(method='ols')
    
    # Print summary
    print(data_handler.generate_summary_report())
    
    # Ask user if they want to proceed if validation fails
    if not (coint_results['is_cointegrated'] and stat_results['is_stationary']):
        response = input("\n⚠️  Statistical tests show mixed results. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n❌ Execution cancelled by user.")
            return
    
    # === 4. GENERATE TRADING SIGNALS ===
    print("\n" + "=" * 70)
    strategy = CrackSpreadStrategy(config.strategy)
    
    try:
        signals = strategy.generate_signals(df, spread)
    except Exception as e:
        print(f"\n❌ Error generating signals: {e}")
        return
    
    # === 5. APPLY RISK MANAGEMENT ===
    print("\n" + "=" * 70)
    risk_manager = RiskManager(config.risk)
    try:
        # FIX: Align df to match signals index
        df_aligned = df.loc[signals.index]  # <-- ADD THIS LINE
        risk_adjusted_signals = risk_manager.apply_risk_filters(df_aligned, signals)  # <-- USE df_aligned
    except Exception as e:
        print(f"\n❌ Error applying risk management: {e}")
        return
        
    # === 6. RUN BACKTEST ===
    print("\n" + "=" * 70)
    backtester = Backtester(config.backtest)
    
    try:
        results = backtester.run_backtest(
            risk_adjusted_signals,
            initial_capital=config.risk.initial_capital
        )
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        return
    
    # === 7. CALCULATE PERFORMANCE METRICS ===
    try:
        metrics = backtester.calculate_performance_metrics()
    except Exception as e:
        print(f"\n❌ Error calculating metrics: {e}")
        return
    
    # === 8. GENERATE VISUALIZATIONS ===
    print("\n" + "=" * 70)
    visualizer = Visualizer(config.output)
    
    try:
        trades_df = backtester.get_trade_dataframe()
        visualizer.create_full_report(df, signals, results, trades_df)
    except Exception as e:
        print(f"\n⚠️  Warning: Could not generate all visualizations: {e}")
    
    # === 9. MONTE CARLO SIMULATION ===
    response = input("\n🎲 Run Monte Carlo simulation? (y/n): ")
    if response.lower() == 'y':
        try:
            mc_results = backtester.monte_carlo_simulation(n_simulations=1000)
            visualizer.plot_monte_carlo_results(
                mc_results['detailed_results'],
                save_path=f"{config.output.results_dir}/08_monte_carlo.{config.output.plot_format}"
            )
            
            # Print summary
            print("\n" + "=" * 70)
            print("🎲 MONTE CARLO SIMULATION SUMMARY")
            print("=" * 70)
            summary = mc_results['summary']
            print(f"Mean Return: {summary['Mean_Return_Pct']:.2f}%")
            print(f"5th Percentile: {summary['Percentile_5_Return']:.2f}%")
            print(f"95th Percentile: {summary['Percentile_95_Return']:.2f}%")
            print(f"Probability of Profit: {summary['Probability_of_Profit']:.1f}%")
            print(f"Risk of 20%+ Loss: {summary['Risk_of_Ruin_20pct']:.1f}%")
            print("=" * 70)
            
        except Exception as e:
            print(f"\n⚠️  Warning: Monte Carlo simulation failed: {e}")
    
    # === 10. EXPORT TRADE JOURNAL ===
    try:
        backtester.generate_trade_journal(
            filepath=f"{config.output.results_dir}/trade_journal.csv"
        )
    except Exception as e:
        print(f"\n⚠️  Warning: Could not export trade journal: {e}")
    
    # === 11. FINAL SUMMARY ===
    print("\n" + "=" * 70)
    print("✅ EXECUTION COMPLETE")
    print("=" * 70)
    print(f"\nFinal Portfolio Value: ${metrics['Final_Portfolio_Value']:,.2f}")
    print(f"Total Return: {metrics['Total_Return_Pct']:.2f}%")
    print(f"CAGR: {metrics['CAGR_Pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
    print(f"Max Drawdown: {metrics['Max_Drawdown_Pct']:.2f}%")
    print(f"Win Rate: {metrics['Win_Rate_Pct']:.1f}%")
    print(f"\nAll results saved to: {config.output.results_dir}/")
    print("=" * 70 + "\n")
    
    return {
        'config': config,
        'data': df,
        'signals': signals,
        'results': results,
        'metrics': metrics,
        'trades': trades_df
    }


def quick_analysis():
    """
    Quick analysis mode - runs strategy with default parameters
    and displays key metrics without full visualization
    """
    print("\n⚡ QUICK ANALYSIS MODE")
    print("=" * 70)
    
    config = Config()
    config.output.save_plots = False
    
    # Fetch data
    data_handler = DataHandler(config.data)
    df = data_handler.fetch_data(verbose=False)
    spread = data_handler.compute_crack_spread()
    
    # Run strategy
    strategy = CrackSpreadStrategy(config.strategy)
    signals = strategy.generate_signals(df, spread)
    
    # Backtest
    backtester = Backtester(config.backtest)
    results = backtester.run_backtest(signals, initial_capital=config.risk.initial_capital)
    metrics = backtester.calculate_performance_metrics()
    
    print("\n✅ Quick analysis complete!")
    return metrics


def optimize_parameters():
    """
    Parameter optimization mode - grid search over key parameters
    """
    print("\n🔧 PARAMETER OPTIMIZATION MODE")
    print("=" * 70)
    
    config = Config()
    
    # Define parameter grid
    param_grid = {
        'window': [20, 30, 40, 60],
        'z_entry_long': [-1.5, -2.0, -2.5],
        'z_entry_short': [1.5, 2.0, 2.5],
    }
    
    # Fetch data once
    data_handler = DataHandler(config.data)
    df = data_handler.fetch_data(verbose=False)
    spread = data_handler.compute_crack_spread()
    
    best_sharpe = -np.inf
    best_params = None
    results_list = []
    
    total_combinations = len(param_grid['window']) * len(param_grid['z_entry_long']) * len(param_grid['z_entry_short'])
    current = 0
    
    for window in param_grid['window']:
        for z_long in param_grid['z_entry_long']:
            for z_short in param_grid['z_entry_short']:
                current += 1
                print(f"\nTesting combination {current}/{total_combinations}: "
                      f"window={window}, z_long={z_long}, z_short={z_short}")
                
                # Update config
                config.strategy.window = window
                config.strategy.z_entry_long = z_long
                config.strategy.z_entry_short = z_short
                
                try:
                    # Run strategy
                    strategy = CrackSpreadStrategy(config.strategy)
                    signals = strategy.generate_signals(df, spread)
                    
                    # Backtest
                    backtester = Backtester(config.backtest)
                    results = backtester.run_backtest(signals)
                    metrics = backtester.calculate_performance_metrics()
                    
                    # Store results
                    results_list.append({
                        'window': window,
                        'z_entry_long': z_long,
                        'z_entry_short': z_short,
                        'sharpe': metrics['Sharpe_Ratio'],
                        'return': metrics['Total_Return_Pct'],
                        'max_dd': metrics['Max_Drawdown_Pct'],
                        'win_rate': metrics['Win_Rate_Pct']
                    })
                    
                    # Track best
                    if metrics['Sharpe_Ratio'] > best_sharpe:
                        best_sharpe = metrics['Sharpe_Ratio']
                        best_params = {
                            'window': window,
                            'z_entry_long': z_long,
                            'z_entry_short': z_short
                        }
                        print(f"  ✅ New best Sharpe: {best_sharpe:.2f}")
                
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    continue
    
    # Display results
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('sharpe', ascending=False)
    
    print("\n" + "=" * 70)
    print("🏆 OPTIMIZATION RESULTS - TOP 5")
    print("=" * 70)
    print(results_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 70)
    print("🎯 BEST PARAMETERS")
    print("=" * 70)
    print(f"Window: {best_params['window']}")
    print(f"Z Entry Long: {best_params['z_entry_long']}")
    print(f"Z Entry Short: {best_params['z_entry_short']}")
    print(f"Best Sharpe Ratio: {best_sharpe:.2f}")
    print("=" * 70)
    
    # Save results
    results_df.to_csv(f"{config.output.results_dir}/optimization_results.csv", index=False)
    print(f"\nFull results saved to: {config.output.results_dir}/optimization_results.csv")
    
    return results_df, best_params


if __name__ == "__main__":
    import sys
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == 'quick':
            quick_analysis()
        elif mode == 'optimize':
            optimize_parameters()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python main.py [quick|optimize]")
            print("  No argument: Run full analysis")
            print("  quick: Quick analysis without visualizations")
            print("  optimize: Parameter optimization grid search")
    else:
        # Run full analysis
        output = main()