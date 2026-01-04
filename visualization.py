"""
Professional visualization suite for pairs trading strategies
Publication-quality charts with institutional aesthetics
Works with any asset pair: stocks, ETFs, futures, commodities, crypto
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


class PairsVisualizer:
    """
    Comprehensive visualization toolkit for pairs trading strategy analysis
    
    Features:
    - Price and spread charts
    - Equity curves with drawdown
    - Trade distribution analysis
    - Performance heatmaps
    - Z-score with signal overlays
    - Monte Carlo simulation results
    - Risk metrics visualization
    - Works with ANY asset pair
    """
    
    def __init__(self, config, pair_name: Optional[str] = None,
                 asset1_name: Optional[str] = None,
                 asset2_name: Optional[str] = None):
        """
        Initialize visualizer
        
        Args:
            config: Configuration object
            pair_name: Descriptive name for the pair (e.g., 'SPY-QQQ')
            asset1_name: Display name for Asset 1 (e.g., 'S&P 500')
            asset2_name: Display name for Asset 2 (e.g., 'Nasdaq 100')
        """
        self.config = config
        self.pair_name = pair_name or config.pair.pair_name
        self.asset1_name = asset1_name or config.pair.asset1_ticker
        self.asset2_name = asset2_name or config.pair.asset2_ticker
        
        self.colors = {
            'primary': '#2E86AB',      # Blue
            'secondary': '#A23B72',    # Purple
            'success': '#06A77D',      # Green
            'danger': '#D81E5B',       # Red
            'warning': '#F77F00',      # Orange
            'neutral': '#264653',      # Dark teal
            'accent': '#E9C46A',       # Gold
        }
    
    def plot_price_series(self, df: pd.DataFrame, 
                         save_path: Optional[str] = None):
        """
        Plot Asset1 and Asset2 price series with volume
        
        Args:
            df: DataFrame with OHLCV data (Asset1/Asset2 columns)
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Asset 1 Price
        axes[0].plot(df.index, df['Asset1_Close'], 
                    label=self.asset1_name, 
                    color=self.colors['primary'], 
                    linewidth=1.5)
        axes[0].set_title(f'{self.asset1_name} Price', fontweight='bold')
        axes[0].set_ylabel('Price ($)', fontweight='bold')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Asset 2 Price
        axes[1].plot(df.index, df['Asset2_Close'], 
                    label=self.asset2_name, 
                    color=self.colors['secondary'], 
                    linewidth=1.5)
        axes[1].set_title(f'{self.asset2_name} Price', fontweight='bold')
        axes[1].set_ylabel('Price ($)', fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Volume comparison
        if 'Asset1_Volume' in df.columns and 'Asset2_Volume' in df.columns:
            axes[2].bar(df.index, df['Asset1_Volume'], 
                       label=f'{self.asset1_name} Volume', 
                       color=self.colors['primary'], 
                       alpha=0.6, width=1)
            axes[2].bar(df.index, df['Asset2_Volume'], 
                       label=f'{self.asset2_name} Volume', 
                       color=self.colors['secondary'], 
                       alpha=0.6, width=1)
            axes[2].set_title('Trading Volume', fontweight='bold')
            axes[2].set_ylabel('Volume', fontweight='bold')
            axes[2].set_xlabel('Date', fontweight='bold')
            axes[2].legend(loc='best')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved price series plot to: {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_spread_analysis(self, signals: pd.DataFrame, 
                            save_path: Optional[str] = None):
        """
        Comprehensive spread analysis with z-score and signals
        
        Args:
            signals: DataFrame with spread, z-score, and trading signals
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # === Spread with Bollinger Bands ===
        axes[0].plot(signals.index, signals['Spread'], 
                    label=f'Price Spread ({self.pair_name})', 
                    color=self.colors['neutral'], 
                    linewidth=1.2)
        axes[0].plot(signals.index, signals['Rolling_Mean'], 
                    label=f'{self.config.strategy.window}-Day Mean', 
                    color=self.colors['danger'], 
                    linewidth=2, 
                    linestyle='--')
        
        # Bollinger Bands
        axes[0].fill_between(signals.index,
                            signals['Rolling_Mean'] + signals['Rolling_Std'],
                            signals['Rolling_Mean'] - signals['Rolling_Std'],
                            alpha=0.2, 
                            color=self.colors['accent'], 
                            label='±1 Std Dev')
        axes[0].fill_between(signals.index,
                            signals['Rolling_Mean'] + 2*signals['Rolling_Std'],
                            signals['Rolling_Mean'] - 2*signals['Rolling_Std'],
                            alpha=0.1, 
                            color=self.colors['accent'], 
                            label='±2 Std Dev')
        
        axes[0].set_title(f'{self.pair_name} Spread with Mean-Reversion Bands', 
                         fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Spread', fontweight='bold')
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        
        # === Z-Score with Thresholds ===
        axes[1].plot(signals.index, signals['Z_Score'], 
                    label='Z-Score', 
                    color=self.colors['primary'], 
                    linewidth=1.2)
        axes[1].axhline(2, color=self.colors['danger'], 
                       linestyle='--', alpha=0.7, linewidth=2, 
                       label='Short Threshold (+2σ)')
        axes[1].axhline(-2, color=self.colors['success'], 
                       linestyle='--', alpha=0.7, linewidth=2, 
                       label='Long Threshold (-2σ)')
        axes[1].axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=1)
        axes[1].fill_between(signals.index, -2, 2, alpha=0.1, color='gray')
        
        # Mark entry signals
        long_entries = signals[signals['Trade_Entry'] & (signals['Position'] > 0)]
        short_entries = signals[signals['Trade_Entry'] & (signals['Position'] < 0)]
        
        axes[1].scatter(long_entries.index, long_entries['Z_Score'], 
                       color=self.colors['success'], marker='^', s=100, 
                       label='Long Entry', zorder=5, edgecolors='black', linewidth=0.5)
        axes[1].scatter(short_entries.index, short_entries['Z_Score'], 
                       color=self.colors['danger'], marker='v', s=100, 
                       label='Short Entry', zorder=5, edgecolors='black', linewidth=0.5)
        
        axes[1].set_title(f'{self.pair_name} Z-Score with Trade Signals', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Z-Score', fontweight='bold')
        axes[1].legend(loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        # === Position Tracking ===
        axes[2].fill_between(signals.index, 0, signals['Position'], 
                            where=signals['Position'] > 0,
                            color=self.colors['success'], alpha=0.5, 
                            label='Long Position')
        axes[2].fill_between(signals.index, 0, signals['Position'], 
                            where=signals['Position'] < 0,
                            color=self.colors['danger'], alpha=0.5, 
                            label='Short Position')
        axes[2].axhline(0, color='black', linestyle='-', linewidth=1)
        
        axes[2].set_title('Position Tracking', fontweight='bold', fontsize=14)
        axes[2].set_ylabel('Position', fontweight='bold')
        axes[2].set_xlabel('Date', fontweight='bold')
        axes[2].legend(loc='best', framealpha=0.9)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved spread analysis plot to: {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_equity_curve(self, results: pd.DataFrame, 
                         save_path: Optional[str] = None):
        """
        Plot equity curve with drawdown
        
        Args:
            results: Backtest results DataFrame
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                gridspec_kw={'height_ratios': [3, 1]})
        
        # === Equity Curve ===
        axes[0].plot(results.index, results['Portfolio_Value'], 
                    label='Portfolio Value', 
                    color=self.colors['primary'], 
                    linewidth=2)
        axes[0].plot(results.index, results['Cumulative_Max'], 
                    label='Peak Value', 
                    color=self.colors['success'], 
                    linewidth=1.5, 
                    linestyle='--', 
                    alpha=0.7)
        
        # Initial capital line
        axes[0].axhline(self.config.initial_capital, 
                       color='gray', linestyle=':', 
                       label='Initial Capital', linewidth=1.5)
        
        # Fill between equity and peak
        axes[0].fill_between(results.index,
                            results['Portfolio_Value'],
                            results['Cumulative_Max'],
                            where=results['Portfolio_Value'] < results['Cumulative_Max'],
                            color=self.colors['danger'],
                            alpha=0.3,
                            label='Drawdown Period')
        
        axes[0].set_title(f'{self.pair_name} Portfolio Equity Curve', fontweight='bold', fontsize=14)
        axes[0].set_ylabel('Portfolio Value ($)', fontweight='bold')
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # === Drawdown ===
        axes[1].fill_between(results.index, 0, results['Drawdown'] * 100,
                            color=self.colors['danger'], alpha=0.6)
        axes[1].set_title('Drawdown', fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Drawdown (%)', fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Annotate max drawdown
        max_dd_idx = results['Drawdown'].idxmin()
        max_dd_val = results['Drawdown'].min() * 100
        axes[1].annotate(f'Max DD: {max_dd_val:.2f}%',
                        xy=(max_dd_idx, max_dd_val),
                        xytext=(10, -30),
                        textcoords='offset points',
                        fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved equity curve plot to: {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_trade_distribution(self, trades_df: pd.DataFrame, 
                               save_path: Optional[str] = None):
        """
        Analyze trade distribution and statistics
        
        Args:
            trades_df: DataFrame with individual trade records
            save_path: Optional path to save figure
        """
        if trades_df.empty:
            print("⚠️  No trades to plot")
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # === 1. P&L Distribution ===
        ax1 = fig.add_subplot(gs[0, :2])
        wins = trades_df[trades_df['Win']]['Portfolio_PnL_Pct']
        losses = trades_df[~trades_df['Win']]['Portfolio_PnL_Pct']
        
        ax1.hist([wins, losses], bins=30, label=['Wins', 'Losses'],
                color=[self.colors['success'], self.colors['danger']],
                alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='black', linestyle='--', linewidth=2)
        ax1.set_title(f'{self.pair_name} Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('P&L (%)', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # === 2. Win Rate Pie Chart ===
        ax2 = fig.add_subplot(gs[0, 2])
        win_rate = (trades_df['Win'].sum() / len(trades_df)) * 100
        sizes = [win_rate, 100 - win_rate]
        colors_pie = [self.colors['success'], self.colors['danger']]
        ax2.pie(sizes, labels=['Wins', 'Losses'], autopct='%1.1f%%',
               colors=colors_pie, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
        ax2.set_title(f'Win Rate: {win_rate:.1f}%', fontweight='bold')
        
        # === 3. Duration Analysis ===
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(trades_df['Duration_Days'], bins=20, 
                color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax3.axvline(trades_df['Duration_Days'].mean(), 
                   color=self.colors['danger'], linestyle='--', linewidth=2,
                   label=f"Mean: {trades_df['Duration_Days'].mean():.1f} days")
        ax3.set_title('Trade Duration', fontweight='bold')
        ax3.set_xlabel('Days', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # === 4. Cumulative P&L ===
        ax4 = fig.add_subplot(gs[1, 1:])
        cumulative_pnl = trades_df['Portfolio_PnL'].cumsum()
        ax4.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl,
                color=self.colors['primary'], linewidth=2, marker='o', markersize=3)
        ax4.axhline(0, color='black', linestyle='--', linewidth=1)
        ax4.fill_between(range(1, len(cumulative_pnl) + 1), 0, cumulative_pnl,
                        where=cumulative_pnl >= 0, color=self.colors['success'], alpha=0.3)
        ax4.fill_between(range(1, len(cumulative_pnl) + 1), 0, cumulative_pnl,
                        where=cumulative_pnl < 0, color=self.colors['danger'], alpha=0.3)
        ax4.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax4.set_xlabel('Trade Number', fontweight='bold')
        ax4.set_ylabel('Cumulative P&L ($)', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # === 5. MAE vs MFE Scatter ===
        ax5 = fig.add_subplot(gs[2, :2])
        colors_scatter = [self.colors['success'] if w else self.colors['danger'] 
                         for w in trades_df['Win']]
        ax5.scatter(trades_df['MAE'], trades_df['MFE'], 
                   c=colors_scatter, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax5.set_title('Maximum Adverse Excursion vs Maximum Favorable Excursion', 
                     fontweight='bold')
        ax5.set_xlabel('MAE (Max Adverse)', fontweight='bold')
        ax5.set_ylabel('MFE (Max Favorable)', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # === 6. Summary Statistics Table ===
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        stats_text = f"""
        TRADE STATISTICS
        ─────────────────────
        Pair: {self.pair_name}
        Total Trades: {len(trades_df)}
        Win Rate: {win_rate:.1f}%
        Avg Win: {wins.mean():.2f}%
        Avg Loss: {losses.mean():.2f}%
        Largest Win: {trades_df['Portfolio_PnL_Pct'].max():.2f}%
        Largest Loss: {trades_df['Portfolio_PnL_Pct'].min():.2f}%
        Avg Duration: {trades_df['Duration_Days'].mean():.1f} days
        Total P&L: ${trades_df['Portfolio_PnL'].sum():,.2f}
        """
        
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved trade distribution plot to: {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_monthly_returns_heatmap(self, results: pd.DataFrame, 
                                    save_path: Optional[str] = None):
        """
        Monthly returns heatmap
        
        Args:
            results: Backtest results DataFrame
            save_path: Optional path to save figure
        """
        # Calculate monthly returns
        monthly_returns = results['Net_Return'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Create pivot table
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        

        pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'}, 
                   linewidths=0.5, linecolor='gray', ax=ax)
        
        ax.set_title(f'{self.pair_name} Monthly Returns Heatmap (%)', 
                     fontweight='bold', fontsize=14, pad=20)
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Year', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved monthly returns heatmap to: {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()

    def plot_rolling_cointegration(self, rolling_coint: pd.DataFrame, 
                                save_path: Optional[str] = None):
        """
        Visualize rolling cointegration analysis
        
        Args:
            rolling_coint: DataFrame from calculate_rolling_cointegration()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: P-value over time
        axes[0].plot(rolling_coint.index, rolling_coint['Coint_PValue'],
                    color=self.colors['primary'], linewidth=1.5, label='P-Value')
        axes[0].axhline(0.05, color=self.colors['danger'], 
                    linestyle='--', linewidth=2, label='Significance Threshold (α=0.05)')
        axes[0].fill_between(rolling_coint.index, 0, 0.05,
                            color=self.colors['success'], alpha=0.2, 
                            label='Cointegrated Zone (p < 0.05)')
        axes[0].set_title(f'{self.pair_name} Rolling Cointegration P-Value Over Time', 
                        fontweight='bold', fontsize=14)
        axes[0].set_ylabel('P-Value', fontweight='bold')
        axes[0].set_ylim([0, max(0.15, rolling_coint['Coint_PValue'].max())])
        axes[0].legend(loc='best', framealpha=0.9)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Binary cointegration status
        axes[1].fill_between(rolling_coint.index, 0, rolling_coint['Is_Cointegrated'],
                            color=self.colors['success'], alpha=0.6, step='post',
                            label='Cointegrated Periods')
        axes[1].fill_between(rolling_coint.index, 
                            rolling_coint['Is_Cointegrated'], 1,
                            where=~rolling_coint['Is_Cointegrated'],
                            color=self.colors['danger'], alpha=0.6, step='post',
                            label='Non-Cointegrated Periods')
        axes[1].set_title('Cointegration Status (Trading Allowed vs. Paused)', 
                        fontweight='bold', fontsize=14)
        axes[1].set_ylabel('Status', fontweight='bold')
        axes[1].set_xlabel('Date', fontweight='bold')
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(['❌ Not Cointegrated', '✅ Cointegrated'])
        axes[1].legend(loc='best', framealpha=0.9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.output.plot_dpi, bbox_inches='tight')
            print(f"✅ Saved rolling cointegration plot to: {save_path}")
        
        if self.config.output.show_plots:
            plt.show()
        else:
            plt.close()