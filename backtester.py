"""
Comprehensive backtesting engine for pairs trading strategies
Supports stocks, ETFs, futures, commodities, and any cointegrated pairs
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


class PairsBacktester:
    """
    Professional-grade backtesting engine for pairs trading
    
    Features:
    - Realistic transaction costs and slippage
    - Position-by-position P&L tracking
    - Walk-forward optimization
    - Monte Carlo simulation
    - Multiple performance metrics
    - Trade journal with detailed analytics
    - Support for any asset pair (stocks, ETFs, futures, etc.)
    """
    
    def __init__(self, config, pair_name: Optional[str] = None):
        """
        Initialize backtester
        
        Args:
            config: Configuration object
            pair_name: Optional descriptive name for the pair
        """
        self.config = config
        self.pair_name = pair_name or config.pair.pair_name
        self.results = None
        self.trades = []
        self.equity_curve = None
        
    def run_backtest(self, signals: pd.DataFrame, 
                    initial_capital: Optional[float] = None) -> pd.DataFrame:
        """
        Execute comprehensive backtest with realistic constraints
        
        Args:
            signals: DataFrame with trading signals and positions
            initial_capital: Starting capital (uses config if None)
        
        Returns:
            DataFrame with complete backtest results
        """
        print("\n" + "=" * 60)
        print(f"🚀 RUNNING BACKTEST: {self.pair_name}")
        print("=" * 60)
        
        capital = initial_capital or self.config.initial_capital
        
        df = signals.copy()
        
        # === 1. Calculate Returns ===
        print("\n1️⃣  Calculating raw returns...")
        df = self._calculate_returns(df)
        
        # === 2. Apply Transaction Costs ===
        print("2️⃣  Applying transaction costs and slippage...")
        df = self._apply_transaction_costs(df)
        
        # === 3. Calculate Portfolio Value ===
        print("3️⃣  Computing portfolio value...")
        df = self._calculate_portfolio_value(df, capital)
        
        # === 4. Calculate Drawdown ===
        print("4️⃣  Calculating drawdown metrics...")
        df = self._calculate_drawdown(df)
        df = self._apply_circuit_breakers(df)
        
        # === 5. Track Individual Trades ===
        print("5️⃣  Extracting trade log...")
        self.trades = self._extract_trades(df)
        
        # === 6. Calculate Rolling Metrics ===
        print("6️⃣  Computing rolling performance metrics...")
        df = self._calculate_rolling_metrics(df)
        
        self.results = df
        self.equity_curve = df['Portfolio_Value']
        
        print("\n✅ Backtest complete!")
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from pair positions using optimal hedge ratio
        
        Works for any asset pair: stocks, ETFs, futures, commodities, crypto
        """
        # Calculate individual asset returns
        df['Asset1_Return'] = df['Asset1_Close'].pct_change()
        df['Asset2_Return'] = df['Asset2_Close'].pct_change()
        
        # Get hedge ratio (default to 1:1 if not available)
        if 'Hedge_Ratio' in df.columns:
            hedge_ratio = df['Hedge_Ratio']
        else:
            hedge_ratio = 1.0
        
        # For long spread: long Asset1, short Asset2 weighted by hedge ratio
        # Position > 0: Buy spread (Asset1 up, Asset2 down = profit)
        # Position < 0: Sell spread (Asset1 down, Asset2 up = profit)
        df['Leg1_Return'] = df['Position'].shift(1) * df['Asset1_Return']
        df['Leg2_Return'] = -df['Position'].shift(1) * hedge_ratio * df['Asset2_Return']
        
        # Gross return is the combined leg performance
        df['Gross_Return'] = df['Leg1_Return'] + df['Leg2_Return']
        
        # Handle NaN values
        df['Gross_Return'].fillna(0, inplace=True)
        
        return df
    
    def _apply_transaction_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply realistic transaction costs and slippage
        
        Costs applied when position changes (entry/exit/scaling)
        """
        # Detect position changes (entries, exits, scaling)
        df['Position_Change'] = df['Position'].diff().fillna(0)
        df['Trade_Occurred'] = (df['Position_Change'] != 0).astype(int)
        
        # Transaction cost = commission + slippage
        total_cost_pct = (
            self.config.backtest.transaction_cost_pct + 
            self.config.backtest.slippage_pct
        )
        
        # Apply costs only when trades occur
        # Cost proportional to size of position change
        df['Transaction_Cost'] = (
            abs(df['Position_Change']) * total_cost_pct * df['Trade_Occurred']
        )
        
        # Futures-specific: Add per-contract commission
        if self.config.backtest.commission_per_contract > 0:
            df['Commission_Dollar'] = (
                abs(df['Position_Change']) * 
                self.config.backtest.commission_per_contract
            )
            # Convert to percentage of portfolio
            df['Transaction_Cost'] += df['Commission_Dollar'] / self.config.initial_capital
        
        # Net return = Gross return - Transaction costs
        df['Net_Return'] = df['Gross_Return'] - df['Transaction_Cost']
        
        # Summary statistics
        total_costs = df['Transaction_Cost'].sum() * 100
        num_trades = df['Trade_Occurred'].sum()
        
        print(f"   Total transaction costs: {total_costs:.4f}%")
        print(f"   Number of transactions: {num_trades}")
        print(f"   Avg cost per transaction: {total_costs/num_trades:.4f}%" if num_trades > 0 else "   Avg cost per transaction: N/A")
        
        return df
    
    def _calculate_portfolio_value(self, df: pd.DataFrame, 
                                   initial_capital: float) -> pd.DataFrame:
        """Calculate portfolio value over time"""
        
        # Cumulative returns (multiplicative)
        df['Cumulative_Return'] = (1 + df['Net_Return']).cumprod()
        
        # Portfolio value
        df['Portfolio_Value'] = initial_capital * df['Cumulative_Return']
        
        # Daily P&L in dollars
        df['Daily_PnL'] = df['Portfolio_Value'].diff().fillna(0)
        
        return df
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drawdown metrics"""
        
        # Running maximum (peak)
        df['Cumulative_Max'] = df['Portfolio_Value'].cummax()
        
        # Drawdown (dollar amount)
        df['Drawdown_Dollar'] = df['Portfolio_Value'] - df['Cumulative_Max']
        
        # Drawdown (percentage)
        df['Drawdown'] = df['Drawdown_Dollar'] / df['Cumulative_Max']
        
        # Underwater period (days since last peak)
        df['Days_Underwater'] = 0
        underwater_count = 0
        
        for i in range(len(df)):
            if df['Drawdown'].iloc[i] < 0:
                underwater_count += 1
            else:
                underwater_count = 0
            df['Days_Underwater'].iloc[i] = underwater_count
        
        return df
    
    def _apply_circuit_breakers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stop trading if catastrophic losses occur"""
        
        for i in range(len(df)):
            # Stop if portfolio goes negative
            if df['Portfolio_Value'].iloc[i] < 0:
                print(f"\n🚨 CIRCUIT BREAKER: Portfolio went negative on {df.index[i]}")
                df.loc[df.index[i]:, 'Position'] = 0
                break
            
            # Stop if drawdown > max allowed
            max_dd_threshold = self.config.risk.max_drawdown_stop if hasattr(self.config.risk, 'max_drawdown_stop') else 0.50
            if df['Drawdown'].iloc[i] < -max_dd_threshold:
                print(f"\n🚨 CIRCUIT BREAKER: {max_dd_threshold*100:.0f}% drawdown hit on {df.index[i]}")
                df.loc[df.index[i]:, 'Position'] = 0
                break
        
        return df
    
    def _extract_trades(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trades for detailed analysis
        
        Returns:
            List of trade dictionaries
        """
        trades = []
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_zscore = None
        direction = None
        entry_portfolio_value = None
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Trade entry
            if row['Trade_Entry'] and not in_trade:
                in_trade = True
                entry_idx = idx
                entry_price = row['Spread']
                entry_zscore = row['Z_Score']
                direction = 'Long' if row['Position'] > 0 else 'Short'
                entry_portfolio_value = row['Portfolio_Value']
            
            # Trade exit
            elif row['Trade_Exit'] and in_trade:
                exit_idx = idx
                exit_price = row['Spread']
                exit_zscore = row['Z_Score']
                exit_portfolio_value = row['Portfolio_Value']
                
                # Calculate P&L
                if direction == 'Long':
                    pnl = exit_price - entry_price
                else:  # Short
                    pnl = entry_price - exit_price
                
                pnl_pct = (pnl / abs(entry_price)) * 100
                
                # Portfolio P&L
                portfolio_pnl = exit_portfolio_value - entry_portfolio_value
                portfolio_pnl_pct = (portfolio_pnl / entry_portfolio_value) * 100
                
                # Calculate MAE and MFE during trade
                trade_slice = df.loc[entry_idx:exit_idx]
                if direction == 'Long':
                    mae = (trade_slice['Spread'] - entry_price).min()
                    mfe = (trade_slice['Spread'] - entry_price).max()
                else:
                    mae = (entry_price - trade_slice['Spread']).min()
                    mfe = (entry_price - trade_slice['Spread']).max()
                
                trades.append({
                    'Trade_Number': len(trades) + 1,
                    'Entry_Date': entry_idx,
                    'Exit_Date': exit_idx,
                    'Direction': direction,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Entry_Z_Score': entry_zscore,
                    'Exit_Z_Score': exit_zscore,
                    'Duration_Days': (exit_idx - entry_idx).days,
                    'Spread_PnL': pnl,
                    'Spread_PnL_Pct': pnl_pct,
                    'Portfolio_PnL': portfolio_pnl,
                    'Portfolio_PnL_Pct': portfolio_pnl_pct,
                    'MAE': mae,  # Maximum Adverse Excursion
                    'MFE': mfe,  # Maximum Favorable Excursion
                    'Win': pnl > 0
                })
                
                in_trade = False
        
        return trades
    
    def _calculate_rolling_metrics(self, df: pd.DataFrame, 
                                   window: int = 60) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        returns = df['Net_Return']
        
        # Rolling Sharpe Ratio
        df['Rolling_Sharpe'] = (
            returns.rolling(window).mean() / 
            returns.rolling(window).std() * 
            np.sqrt(252)
        )
        
        # Rolling volatility
        df['Rolling_Volatility'] = (
            returns.rolling(window).std() * np.sqrt(252) * 100
        )
        
        # Rolling win rate
        df['Rolling_Win_Rate'] = (
            (returns > 0).rolling(window).mean() * 100
        )
        
        return df
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Returns:
            Dictionary with all performance statistics
        """
        if self.results is None:
            raise ValueError("Run backtest first")
        
        print("\n" + "=" * 60)
        print(f"📊 PERFORMANCE METRICS: {self.pair_name}")
        print("=" * 60)
        
        df = self.results
        returns = df['Net_Return'].dropna()
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # === Return Metrics ===
        total_return = (df['Portfolio_Value'].iloc[-1] / self.config.initial_capital - 1) * 100
        
        n_years = len(df) / 252
        cagr = ((df['Portfolio_Value'].iloc[-1] / self.config.initial_capital) ** (1 / n_years) - 1) * 100
        
        ann_return = returns.mean() * 252 * 100
        ann_volatility = returns.std() * np.sqrt(252) * 100
        
        # === Risk-Adjusted Metrics ===
        sharpe_ratio = self._sharpe_ratio(returns)
        sortino_ratio = self._sortino_ratio(returns)
        calmar_ratio = self._calmar_ratio(returns, df['Drawdown'])
        
        # === Drawdown Metrics ===
        max_dd = df['Drawdown'].min() * 100
        max_dd_duration = df['Days_Underwater'].max()
        avg_dd = df[df['Drawdown'] < 0]['Drawdown'].mean() * 100 if len(df[df['Drawdown'] < 0]) > 0 else 0
        
        # === Trade Metrics ===
        if len(trades_df) > 0:
            wins = trades_df[trades_df['Win']]
            losses = trades_df[~trades_df['Win']]
            
            total_trades = len(trades_df)
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
            
            avg_win = wins['Portfolio_PnL_Pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['Portfolio_PnL_Pct'].mean() if len(losses) > 0 else 0
            
            largest_win = trades_df['Portfolio_PnL_Pct'].max() if total_trades > 0 else 0
            largest_loss = trades_df['Portfolio_PnL_Pct'].min() if total_trades > 0 else 0
            
            avg_duration = trades_df['Duration_Days'].mean()
            
            # Profit factor
            gross_profit = wins['Portfolio_PnL'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['Portfolio_PnL'].sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Expectancy
            expectancy = trades_df['Portfolio_PnL'].mean()
            
            # Consecutive wins/losses
            consecutive_wins = self._max_consecutive_wins(trades_df)
            consecutive_losses = self._max_consecutive_losses(trades_df)
            
        else:
            total_trades = win_rate = avg_win = avg_loss = 0
            largest_win = largest_loss = avg_duration = 0
            profit_factor = expectancy = 0
            consecutive_wins = consecutive_losses = 0
        
        # === Exposure Metrics ===
        days_in_market = (df['Position'] != 0).sum()
        pct_in_market = (days_in_market / len(df)) * 100
        
        metrics = {
            # Return metrics
            'Total_Return_Pct': total_return,
            'CAGR_Pct': cagr,
            'Annualized_Return_Pct': ann_return,
            'Annualized_Volatility_Pct': ann_volatility,
            
            # Risk-adjusted returns
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            
            # Drawdown metrics
            'Max_Drawdown_Pct': max_dd,
            'Avg_Drawdown_Pct': avg_dd,
            'Max_Drawdown_Duration_Days': max_dd_duration,
            
            # Trade metrics
            'Total_Trades': total_trades,
            'Win_Rate_Pct': win_rate,
            'Avg_Win_Pct': avg_win,
            'Avg_Loss_Pct': avg_loss,
            'Largest_Win_Pct': largest_win,
            'Largest_Loss_Pct': largest_loss,
            'Avg_Trade_Duration_Days': avg_duration,
            'Profit_Factor': profit_factor,
            'Expectancy': expectancy,
            'Max_Consecutive_Wins': consecutive_wins,
            'Max_Consecutive_Losses': consecutive_losses,
            
            # Exposure
            'Days_In_Market': days_in_market,
            'Market_Exposure_Pct': pct_in_market,
            
            # Final values
            'Initial_Capital': self.config.initial_capital,
            'Final_Portfolio_Value': df['Portfolio_Value'].iloc[-1],
            'Total_PnL': df['Portfolio_Value'].iloc[-1] - self.config.initial_capital,
        }
        
        self._print_metrics_table(metrics)
        
        return metrics
    
    def _sharpe_ratio(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Sharpe Ratio (annualized)"""
        excess_returns = returns - rf_rate / 252
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0
    
    def _sortino_ratio(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Sortino Ratio (downside deviation)"""
        excess_returns = returns - rf_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std()
        return (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    
    def _calmar_ratio(self, returns: pd.Series, drawdown: pd.Series) -> float:
        """Calmar Ratio (return/max drawdown)"""
        ann_return = returns.mean() * 252
        max_dd = abs(drawdown.min())
        return ann_return / max_dd if max_dd != 0 else 0
    
    def _max_consecutive_wins(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive wins"""
        if len(trades_df) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for win in trades_df['Win']:
            if win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _max_consecutive_losses(self, trades_df: pd.DataFrame) -> int:
        """Calculate maximum consecutive losses"""
        if len(trades_df) == 0:
            return 0
        
        max_streak = 0
        current_streak = 0
        
        for win in trades_df['Win']:
            if not win:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _print_metrics_table(self, metrics: Dict):
        """Pretty print metrics in organized table"""
        
        print("\n" + "─" * 60)
        print("📈 RETURN METRICS")
        print("─" * 60)
        print(f"Total Return:              {metrics['Total_Return_Pct']:>12.2f}%")
        print(f"CAGR:                      {metrics['CAGR_Pct']:>12.2f}%")
        print(f"Annualized Return:         {metrics['Annualized_Return_Pct']:>12.2f}%")
        print(f"Annualized Volatility:     {metrics['Annualized_Volatility_Pct']:>12.2f}%")
        
        print("\n" + "─" * 60)
        print("⚖️  RISK-ADJUSTED RETURNS")
        print("─" * 60)
        print(f"Sharpe Ratio:              {metrics['Sharpe_Ratio']:>12.2f}")
        print(f"Sortino Ratio:             {metrics['Sortino_Ratio']:>12.2f}")
        print(f"Calmar Ratio:              {metrics['Calmar_Ratio']:>12.2f}")
        
        print("\n" + "─" * 60)
        print("📉 DRAWDOWN METRICS")
        print("─" * 60)
        print(f"Max Drawdown:              {metrics['Max_Drawdown_Pct']:>12.2f}%")
        print(f"Avg Drawdown:              {metrics['Avg_Drawdown_Pct']:>12.2f}%")
        print(f"Max DD Duration:           {metrics['Max_Drawdown_Duration_Days']:>12.0f} days")
        
        print("\n" + "─" * 60)
        print("💼 TRADE STATISTICS")
        print("─" * 60)
        print(f"Total Trades:              {metrics['Total_Trades']:>12.0f}")
        print(f"Win Rate:                  {metrics['Win_Rate_Pct']:>12.2f}%")
        print(f"Avg Win:                   {metrics['Avg_Win_Pct']:>12.2f}%")
        print(f"Avg Loss:                  {metrics['Avg_Loss_Pct']:>12.2f}%")
        print(f"Largest Win:               {metrics['Largest_Win_Pct']:>12.2f}%")
        print(f"Largest Loss:              {metrics['Largest_Loss_Pct']:>12.2f}%")
        print(f"Profit Factor:             {metrics['Profit_Factor']:>12.2f}")
        print(f"Expectancy:                ${metrics['Expectancy']:>11,.2f}")
        print(f"Avg Trade Duration:        {metrics['Avg_Trade_Duration_Days']:>12.1f} days")
        print(f"Max Consecutive Wins:      {metrics['Max_Consecutive_Wins']:>12.0f}")
        print(f"Max Consecutive Losses:    {metrics['Max_Consecutive_Losses']:>12.0f}")
        
        print("\n" + "─" * 60)
        print("🎯 EXPOSURE METRICS")
        print("─" * 60)
        print(f"Days in Market:            {metrics['Days_In_Market']:>12.0f}")
        print(f"Market Exposure:           {metrics['Market_Exposure_Pct']:>12.2f}%")
        
        print("\n" + "─" * 60)
        print("💰 PORTFOLIO SUMMARY")
        print("─" * 60)
        print(f"Initial Capital:           ${metrics['Initial_Capital']:>11,.2f}")
        print(f"Final Value:               ${metrics['Final_Portfolio_Value']:>11,.2f}")
        print(f"Total P&L:                 ${metrics['Total_PnL']:>11,.2f}")
        
        print("=" * 60 + "\n")


    def get_trade_dataframe(self) -> pd.DataFrame:
        """
        Return trades as DataFrame for easy analysis
        
        Returns:
            pd.DataFrame: Trades with all metadata
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)

    def generate_trade_journal(self, filepath: str = 'results/trade_journal.csv'):
        """
        Export detailed trade journal to CSV
        
        Args:
            filepath: Path to save CSV file
        """
        if not self.trades:
            print("⚠️  No trades to export")
            return
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(filepath, index=False)
        print(f"\n✅ Trade journal exported to: {filepath}")
        print(f"   Total trades: {len(trades_df)}")