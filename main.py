"""
Universal Pairs Trading Strategy - Main Execution Script

This is the orchestration script for a professional-grade pairs trading system
that works with ANY cointegrated asset pair: stocks, ETFs, futures, commodities, crypto.

Author: Alexander Robbins
University of Florida
"""

import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all components
from config import Config
from data_handler import PairsDataHandler
from strategy import PairsMeanReversionStrategy
from backtester import PairsBacktester
from risk_manager import PairsRiskManager
from visualization import PairsVisualizer


def print_banner():
    """Print welcome banner"""
    print("\n" + "=" * 70)
    print("  🚀 UNIVERSAL PAIRS TRADING SYSTEM")
    print("  Professional-Grade Mean-Reversion Strategy")
    print("  University of Florida - Quantitative Finance")
    print("=" * 70)
    print(f"  Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def run_pairs_strategy(asset1: str, 
                       asset2: str,
                       pair_name: str,
                       start_date: str = '2015-01-01',
                       initial_capital: float = 500_000,
                       asset_type: str = 'stock',
                       save_results: bool = True,
                       show_plots: bool = True) -> dict:
    """
    Complete end-to-end pairs trading pipeline
    
    Args:
        asset1: First asset ticker (e.g., 'SPY', 'GLD', 'CL=F')
        asset2: Second asset ticker (e.g., 'QQQ', 'SLV', 'HO=F')
        pair_name: Descriptive name (e.g., 'SPY-QQQ', 'Gold-Silver')
        start_date: Backtest start date
        initial_capital: Starting capital
        asset_type: 'stock', 'etf', 'futures', 'crypto'
        save_results: Save plots and reports
        show_plots: Display plots interactively
    
    Returns:
        dict: Complete results including metrics, trades, and figures
    """
    
    print("\n" + "=" * 70)
    print(f"📊 ANALYZING PAIR: {pair_name}")
    print("=" * 70)
    print(f"Asset 1: {asset1}")
    print(f"Asset 2: {asset2}")
    print(f"Period: {start_date} to present")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: CONFIGURATION
    # ========================================================================
    print("\n🔧 Step 1/7: Initializing configuration...")
    config = Config(
        asset1=asset1,
        asset2=asset2,
        pair_name=pair_name,
        start_date=start_date,
        initial_capital=initial_capital
    )
    config.output.show_plots = show_plots
    config.output.save_plots = save_results
    
    print(config.summary())
    
    # ========================================================================
    # STEP 2: DATA ACQUISITION & VALIDATION
    # ========================================================================
    print("\n📥 Step 2/7: Fetching and validating data...")
    handler = PairsDataHandler(config, asset1, asset2, pair_name)
    
    try:
        df = handler.fetch_data(verbose=True)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to fetch data: {e}")
        return {'error': str(e)}
    
    # Calculate spread
    spread = handler.compute_spread(method='log')
    
    # Statistical validation
    handler.test_cointegration(verbose=True)
    handler.test_stationarity(spread, verbose=True)
    handler.calculate_half_life(spread, verbose=True)
    handler.calculate_hedge_ratio(method='ols')
    
    # Print validation summary
    print(handler.generate_summary_report())
    
    # Optional: Rolling cointegration analysis
    print("\n🔄 Computing rolling cointegration...")
    rolling_coint = handler.calculate_rolling_cointegration(window=252)
    
    # ========================================================================
    # STEP 3: SIGNAL GENERATION
    # ========================================================================
    print("\n📈 Step 3/7: Generating trading signals...")
    strategy = PairsMeanReversionStrategy(config, pair_name=pair_name)
    
    try:
        signals = strategy.generate_signals(df, spread, rolling_coint=rolling_coint)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate signals: {e}")
        return {'error': str(e)}
    
    # ========================================================================
    # STEP 4: RISK MANAGEMENT
    # ========================================================================
    print("\n🛡️  Step 4/7: Applying risk management filters...")
    risk_manager = PairsRiskManager(config, pair_name=pair_name)
    
    try:
        risk_adjusted_signals = risk_manager.apply_risk_filters(
            df, 
            signals,
            asset_type=asset_type
        )
    except Exception as e:
        print(f"\n⚠️  WARNING: Risk management failed: {e}")
        print("    Proceeding with unfiltered signals...")
        risk_adjusted_signals = signals
    
    # ========================================================================
    # STEP 5: BACKTESTING
    # ========================================================================
    print("\n🚀 Step 5/7: Running comprehensive backtest...")
    backtester = PairsBacktester(config, pair_name=pair_name)
    
    try:
        results = backtester.run_backtest(
            risk_adjusted_signals,
            initial_capital=initial_capital
        )
        
        # Calculate performance metrics
        metrics = backtester.calculate_performance_metrics()
        
        # Get trade list
        trades_df = pd.DataFrame(backtester.trades) if backtester.trades else pd.DataFrame()        
    except Exception as e:
        print(f"\n❌ ERROR: Backtest failed: {e}")
        return {'error': str(e)}
    
    # ========================================================================
    # STEP 6: VISUALIZATION
    # ========================================================================
    print("\n📊 Step 6/7: Generating visualizations...")
    viz = PairsVisualizer(config, pair_name=pair_name,
                         asset1_name=asset1, asset2_name=asset2)
    
    if save_results:
        output_dir = f"{config.output.results_dir}/{pair_name.replace(' ', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"   Saving results to: {output_dir}/")
        
        # Generate all plots
        viz.plot_price_series(df, save_path=f"{output_dir}/01_prices.png")
        viz.plot_spread_analysis(signals, save_path=f"{output_dir}/02_spread.png")
        viz.plot_equity_curve(results, save_path=f"{output_dir}/03_equity.png")
        
        if not trades_df.empty:
            viz.plot_trade_distribution(trades_df, save_path=f"{output_dir}/04_trades.png")
        
        viz.plot_monthly_returns_heatmap(results, save_path=f"{output_dir}/05_monthly.png")
        viz.plot_rolling_cointegration(rolling_coint, save_path=f"{output_dir}/06_cointegration.png")
        
        # Export trade journal
        backtester.generate_trade_journal(f"{output_dir}/trade_journal.csv")
        
        print(f"\n✅ All results saved to: {output_dir}/")
    
    # ========================================================================
    # STEP 7: SUMMARY REPORT
    # ========================================================================
    print("\n📋 Step 7/7: Generating summary report...")
    
    print("\n" + "=" * 70)
    print(f"🎯 FINAL SUMMARY: {pair_name}")
    print("=" * 70)
    
    # Key metrics
    print(f"\n💰 PERFORMANCE:")
    print(f"   Total Return:        {metrics['Total_Return_Pct']:>10.2f}%")
    print(f"   CAGR:                {metrics['CAGR_Pct']:>10.2f}%")
    print(f"   Sharpe Ratio:        {metrics['Sharpe_Ratio']:>10.2f}")
    print(f"   Max Drawdown:        {metrics['Max_Drawdown_Pct']:>10.2f}%")
    
    print(f"\n📊 TRADING:")
    print(f"   Total Trades:        {metrics['Total_Trades']:>10.0f}")
    print(f"   Win Rate:            {metrics['Win_Rate_Pct']:>10.2f}%")
    print(f"   Profit Factor:       {metrics['Profit_Factor']:>10.2f}")
    
    print(f"\n💵 CAPITAL:")
    print(f"   Initial Capital:     ${metrics['Initial_Capital']:>10,.0f}")
    print(f"   Final Value:         ${metrics['Final_Portfolio_Value']:>10,.0f}")
    print(f"   Total P&L:           ${metrics['Total_PnL']:>10,.0f}")
    
    # Assessment
    print(f"\n🎯 ASSESSMENT:")
    if metrics['Sharpe_Ratio'] > 1.5 and metrics['Win_Rate_Pct'] > 55:
        print("   ✅ EXCELLENT - Strong performance with good risk-adjusted returns")
    elif metrics['Sharpe_Ratio'] > 1.0 and metrics['Win_Rate_Pct'] > 50:
        print("   ✅ GOOD - Solid performance, acceptable risk profile")
    elif metrics['Sharpe_Ratio'] > 0.5:
        print("   ⚠️  MARGINAL - Needs optimization or alternative parameters")
    else:
        print("   ❌ POOR - Consider different pair or strategy adjustments")
    
    print("=" * 70)
    
    # Return complete results package
    return {
        'config': config,
        'data': df,
        'spread': spread,
        'signals': signals,
        'results': results,
        'metrics': metrics,
        'trades': trades_df,
        'validation': handler.validation_results,
        'pair_name': pair_name
    }


def run_multiple_pairs(pairs_list: list, **kwargs) -> pd.DataFrame:
    """
    Run strategy on multiple pairs and compare results
    
    Args:
        pairs_list: List of tuples [(asset1, asset2, pair_name), ...]
        **kwargs: Additional arguments passed to run_pairs_strategy
    
    Returns:
        DataFrame with comparative metrics
    """
    print("\n" + "=" * 70)
    print("🔍 MULTI-PAIR COMPARISON ANALYSIS")
    print("=" * 70)
    print(f"Analyzing {len(pairs_list)} pairs...")
    
    results_summary = []
    
    for i, (asset1, asset2, pair_name) in enumerate(pairs_list, 1):
        print(f"\n\n{'='*70}")
        print(f"PAIR {i}/{len(pairs_list)}: {pair_name}")
        print(f"{'='*70}")
        
        try:
            result = run_pairs_strategy(
                asset1=asset1,
                asset2=asset2,
                pair_name=pair_name,
                show_plots=False,  # Don't show plots for batch processing
                **kwargs
            )
            
            if 'error' not in result:
                metrics = result['metrics']
                validation = result['validation']
                
                results_summary.append({
                    'Pair': pair_name,
                    'Asset1': asset1,
                    'Asset2': asset2,
                    'Total_Return_%': metrics['Total_Return_Pct'],
                    'CAGR_%': metrics['CAGR_Pct'],
                    'Sharpe_Ratio': metrics['Sharpe_Ratio'],
                    'Max_DD_%': metrics['Max_Drawdown_Pct'],
                    'Win_Rate_%': metrics['Win_Rate_Pct'],
                    'Total_Trades': metrics['Total_Trades'],
                    'Profit_Factor': metrics['Profit_Factor'],
                    'Cointegrated': validation.get('is_cointegrated', False),
                    'Half_Life': validation.get('half_life', np.inf)
                })
            else:
                print(f"⚠️  Skipping {pair_name} due to error: {result['error']}")
        
        except Exception as e:
            print(f"❌ ERROR processing {pair_name}: {e}")
            continue
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results_summary)
    
    if len(comparison_df) > 0:
        # Sort by Sharpe Ratio
        comparison_df = comparison_df.sort_values('Sharpe_Ratio', ascending=False)
        
        print("\n\n" + "=" * 70)
        print("📊 COMPARATIVE RESULTS")
        print("=" * 70)
        print(comparison_df.to_string(index=False))
        print("=" * 70)
        
        # Save to CSV
        output_file = f"results/multi_pair_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        comparison_df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")
    
    return comparison_df


# ============================================================================
# EXAMPLE USE CASES
# ============================================================================

def example_1_spy_qqq():
    """Example 1: S&P 500 vs Nasdaq ETFs"""
    print("\n🎯 EXAMPLE 1: SPY-QQQ (Equity Index Pair)")
    return run_pairs_strategy(
        asset1='SPY',
        asset2='QQQ',
        pair_name='SPY-QQQ',
        start_date='2015-01-01',
        initial_capital=500_000,
        asset_type='etf'
    )


def example_2_gold_silver():
    """Example 2: Gold vs Silver"""
    print("\n🎯 EXAMPLE 2: GLD-SLV (Precious Metals)")
    return run_pairs_strategy(
        asset1='GLD',
        asset2='SLV',
        pair_name='Gold-Silver',
        start_date='2015-01-01',
        initial_capital=500_000,
        asset_type='etf'
    )


def example_3_oil_crack_spread():
    """Example 3: Original CL-HO crack spread (futures)"""
    print("\n🎯 EXAMPLE 3: CL-HO (Oil Crack Spread - Futures)")
    return run_pairs_strategy(
        asset1='CL=F',
        asset2='HO=F',
        pair_name='Crude-HeatingOil',
        start_date='2010-01-01',
        initial_capital=500_000,
        asset_type='futures'
    )


def example_4_tech_stocks():
    """Example 4: Tech stock pair"""
    print("\n🎯 EXAMPLE 4: AAPL-MSFT (Tech Giants)")
    return run_pairs_strategy(
        asset1='AAPL',
        asset2='MSFT',
        pair_name='Apple-Microsoft',
        start_date='2018-01-01',
        initial_capital=500_000,
        asset_type='stock'
    )


def example_5_multi_pair_comparison():
    """Example 5: Compare multiple pairs"""
    print("\n🎯 EXAMPLE 5: Multi-Pair Portfolio Analysis")
    
    pairs = [
        ('SPY', 'QQQ', 'SPY-QQQ'),
        ('GLD', 'SLV', 'Gold-Silver'),
        ('GLD', 'GDX', 'Gold-Miners'),
        ('EFA', 'EEM', 'Developed-Emerging'),
        ('XLE', 'XLF', 'Energy-Finance')
    ]
    
    return run_multiple_pairs(
        pairs,
        start_date='2015-01-01',
        initial_capital=500_000,
        asset_type='etf',
        save_results=True
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with user interaction"""
    
    print_banner()
    
    print("Select an example to run:")
    print("\n1. SPY-QQQ (S&P 500 vs Nasdaq ETFs)")
    print("2. GLD-SLV (Gold vs Silver)")
    print("3. CL-HO (Oil Crack Spread - Original Strategy)")
    print("4. AAPL-MSFT (Tech Giants)")
    print("5. Multi-Pair Comparison (All of the above)")
    print("6. Custom Pair (Enter your own)")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            result = example_1_spy_qqq()
        elif choice == '2':
            result = example_2_gold_silver()
        elif choice == '3':
            result = example_3_oil_crack_spread()
        elif choice == '4':
            result = example_4_tech_stocks()
        elif choice == '5':
            result = example_5_multi_pair_comparison()
        elif choice == '6':
            # Custom pair
            print("\n🎯 CUSTOM PAIR ANALYSIS")
            asset1 = input("Enter Asset 1 ticker (e.g., SPY): ").strip().upper()
            asset2 = input("Enter Asset 2 ticker (e.g., QQQ): ").strip().upper()
            pair_name = input("Enter pair name (e.g., SPY-QQQ): ").strip()
            start_date = input("Enter start date (YYYY-MM-DD, default 2015-01-01): ").strip() or '2015-01-01'
            
            result = run_pairs_strategy(
                asset1=asset1,
                asset2=asset2,
                pair_name=pair_name,
                start_date=start_date,
                initial_capital=500_000,
                asset_type='stock'
            )
        elif choice == '0':
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please run again.")
            sys.exit(1)
        
        print("\n\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)
        print("\nThank you for using the Universal Pairs Trading System!")
        print("For questions or support, contact: alexander.robbins@ufl.edu")
        print("=" * 70 + "\n")
        
        return result
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()