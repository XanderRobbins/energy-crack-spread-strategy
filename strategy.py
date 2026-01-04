"""
Universal mean-reversion strategy for pairs trading
Supports stocks, ETFs, futures, commodities, and any cointegrated pairs
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


class PairsMeanReversionStrategy:
    """
    Sophisticated mean-reversion strategy for cointegrated pairs
    
    Features:
    - Dynamic z-score thresholds based on volatility regime
    - Market regime detection (mean-reverting vs trending)
    - Multiple entry/exit conditions
    - Position pyramiding support
    - Adaptive lookback windows
    - Momentum filters to avoid catching falling knives
    - Optional rolling cointegration filter
    
    Works with any pair: stocks, ETFs, futures, commodities, crypto
    """
    
    def __init__(self, config, pair_name: Optional[str] = None):
        """
        Initialize strategy
        
        Args:
            config: Configuration object
            pair_name: Optional name for the pair (e.g., 'SPY-QQQ')
        """
        self.config = config
        self.pair_name = pair_name or config.pair.pair_name
        self.signals = None
        self.spread = None
        self.trade_log = []
        
    def generate_signals(self, df: pd.DataFrame, spread: pd.Series,
                        rolling_coint: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate trading signals with enhanced logic
        
        Args:
            df: DataFrame with OHLCV data for both assets
            spread: Pre-computed spread series (typically log spread)
            rolling_coint: Optional rolling cointegration results for filtering
        
        Returns:
            DataFrame with signals and intermediate calculations
        """
        print("\n" + "=" * 60)
        print(f"📈 GENERATING TRADING SIGNALS: {self.pair_name}")
        print("=" * 60)
        
        signals = df.copy()
        signals['Spread'] = spread
        self.spread = spread
        
        # === 1. Core Mean-Reversion Indicators ===
        print("\n1️⃣  Calculating z-score indicators...")
        signals = self._calculate_zscore(signals)
        
        # === 2. Regime Detection ===
        if self.config.strategy.use_regime_filter:
            print("2️⃣  Detecting market regime...")
            signals = self._detect_regime(signals)
        else:
            signals['Regime'] = 'Neutral'
        
        # === 3. Volatility Adjustment ===
        if self.config.strategy.use_dynamic_thresholds:
            print("3️⃣  Adjusting thresholds for volatility...")
            signals = self._adjust_thresholds(signals)
        else:
            signals['Z_Threshold_Long'] = self.config.strategy.z_entry_long
            signals['Z_Threshold_Short'] = self.config.strategy.z_entry_short
        
        # === 4. Momentum Filter ===
        print("4️⃣  Adding momentum filters...")
        signals = self._add_momentum_filter(signals)
        
        # === 5. Generate Entry Signals ===
        print("5️⃣  Generating entry signals...")
        signals = self._generate_entry_signals(signals, rolling_coint)
        
        # === 6. Generate Exit Signals ===
        print("6️⃣  Generating exit signals...")
        signals = self._generate_exit_signals(signals)
        
        # === 7. Position Management ===
        print("7️⃣  Managing positions...")
        signals = self._manage_positions(signals)
        
        # === 8. Add Trade Metadata ===
        signals = self._add_trade_metadata(signals)
        
        # Summary statistics
        self._print_signal_summary(signals)
        
        self.signals = signals.dropna()
        return self.signals
    
    def _calculate_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling z-score of spread with multiple lookback windows"""
        
        # Primary z-score (trading signal)
        df['Rolling_Mean'] = df['Spread'].rolling(
            window=self.config.strategy.window,
            min_periods=self.config.strategy.window
        ).mean()
        
        df['Rolling_Std'] = df['Spread'].rolling(
            window=self.config.strategy.window,
            min_periods=self.config.strategy.window
        ).std()
        
        # Z-score: (current - mean) / std
        df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']
        
        # Long-term mean for regime detection
        df['LT_Mean'] = df['Spread'].rolling(
            window=self.config.strategy.lookback_period,
            min_periods=self.config.strategy.lookback_period // 2
        ).mean()
        
        # Long-term std for comparison
        df['LT_Std'] = df['Spread'].rolling(
            window=self.config.strategy.lookback_period,
            min_periods=self.config.strategy.lookback_period // 2
        ).std()
        
        return df
    
    def _detect_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regime: mean-reverting vs trending
        Uses moving average crossovers and volatility expansion
        """
        # Short and long-term MAs of spread
        df['SMA_Fast'] = df['Spread'].rolling(self.config.strategy.trend_sma_fast).mean()
        df['SMA_Slow'] = df['Spread'].rolling(self.config.strategy.trend_sma_slow).mean()
        
        # Regime classification based on trend strength
        conditions = [
            df['SMA_Fast'] > df['SMA_Slow'] * 1.02,  # Strong uptrend (2% divergence)
            df['SMA_Fast'] < df['SMA_Slow'] * 0.98,  # Strong downtrend
        ]
        
        choices = ['Trending_Up', 'Trending_Down']
        df['Regime'] = np.select(conditions, choices, default='Mean_Reverting')
        
        # Volatility regime (expanding = trending, contracting = mean-reverting)
        df['Vol_Ratio'] = df['Rolling_Std'] / df['LT_Std']
        df['High_Vol'] = df['Vol_Ratio'] > 1.2  # 20% above long-term average
        
        # Combine trend and volatility into composite regime
        df['Regime_Composite'] = df['Regime']
        df.loc[df['High_Vol'] & (df['Regime'] != 'Mean_Reverting'), 'Regime_Composite'] = 'Volatile_Trending'
        
        return df
    
    def _adjust_thresholds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamically adjust z-score thresholds based on volatility regime
        
        Logic:
        - Low volatility: Use tighter thresholds (more trades)
        - High volatility: Use wider thresholds (fewer, higher quality trades)
        """
        # Calculate volatility ratio
        vol_ratio = df['Rolling_Std'] / df['Rolling_Std'].rolling(
            self.config.strategy.vol_lookback
        ).mean()
        
        # Adjust thresholds (inverse relationship with volatility)
        # When vol is high (ratio > 1), make thresholds more extreme
        # When vol is low (ratio < 1), make thresholds tighter
        df['Z_Threshold_Long'] = self.config.strategy.z_entry_long * vol_ratio
        df['Z_Threshold_Short'] = self.config.strategy.z_entry_short * vol_ratio
        
        # Clamp thresholds to reasonable bounds
        df['Z_Threshold_Long'] = df['Z_Threshold_Long'].clip(lower=-3.5, upper=-1.5)
        df['Z_Threshold_Short'] = df['Z_Threshold_Short'].clip(lower=1.5, upper=3.5)
        
        return df
    
    def _add_momentum_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators to filter out weak signals
        Prevents entering trades against strong momentum
        """
        # Rate of change of spread
        df['Spread_ROC'] = df['Spread'].pct_change(5)  # 5-day ROC
        
        # Rate of change of z-score (momentum of mean reversion)
        df['Z_Momentum'] = df['Z_Score'].diff(3)  # 3-day change
        
        # Spread acceleration (second derivative)
        df['Spread_Accel'] = df['Spread_ROC'].diff(1)
        
        # Momentum confirmation flags
        df['Long_Momentum_OK'] = (
            (df['Z_Momentum'] < 0) |  # Z-score declining (getting more extreme)
            (df['Spread_Accel'] < 0)   # Spread decelerating downward
        )
        
        df['Short_Momentum_OK'] = (
            (df['Z_Momentum'] > 0) |  # Z-score rising (getting more extreme)
            (df['Spread_Accel'] > 0)   # Spread decelerating upward
        )
        
        return df
    
    def _generate_entry_signals(self, df: pd.DataFrame,
                                rolling_coint: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate entry signals with optional cointegration filter
        
        Entry logic:
        - Long: Z-score < threshold, mean-reverting regime, good momentum
        - Short: Z-score > threshold, mean-reverting regime, good momentum
        - Optional: Only trade when pair is cointegrated
        """
        df['Signal'] = 0
        
        # Base entry conditions
        long_zscore = df['Z_Score'] < df['Z_Threshold_Long']
        long_regime = df['Regime_Composite'].isin(['Mean_Reverting', 'Trending_Down'])
        long_momentum = df['Long_Momentum_OK']
        
        short_zscore = df['Z_Score'] > df['Z_Threshold_Short']
        short_regime = df['Regime_Composite'].isin(['Mean_Reverting', 'Trending_Up'])
        short_momentum = df['Short_Momentum_OK']
        
        long_condition = long_zscore & long_regime & long_momentum
        short_condition = short_zscore & short_regime & short_momentum
        
        # Apply cointegration filter if available
        if rolling_coint is not None and not rolling_coint.empty:
            print("   ✅ Applying rolling cointegration filter...")
            try:
                coint_status = rolling_coint['Is_Cointegrated'].reindex(
                    df.index, method='ffill'
                ).fillna(False)
                
                long_condition = long_condition & coint_status
                short_condition = short_condition & coint_status
                
                cointegrated_days = coint_status.sum()
                total_days = len(df)
                pct_coint = (cointegrated_days / total_days * 100) if total_days > 0 else 0
                print(f"   Trading on {cointegrated_days}/{total_days} days ({pct_coint:.1f}%)")
            except Exception as e:
                print(f"   ⚠️ Warning: Could not apply cointegration filter: {e}")
                print("   Proceeding without cointegration filter...")
        else:
            print("   ℹ️  No cointegration filter applied (all periods eligible)")
        
        # Set signals
        df.loc[long_condition, 'Signal'] = 1
        df.loc[short_condition, 'Signal'] = -1
        
        return df
    
    def _generate_exit_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate exit signals based on multiple criteria
        
        Exit conditions:
        1. Z-score crosses back to mean (primary exit)
        2. Z-score reverses direction (stop loss)
        3. Regime changes to unfavorable (risk management)
        """
        # Primary exit: z-score crosses exit threshold
        df['Exit_Mean_Reversion'] = abs(df['Z_Score']) < self.config.strategy.z_exit
        
        # Regime change exit
        df['Exit_Regime_Change'] = df['Regime_Composite'] == 'Volatile_Trending'
        
        return df
    
    def _manage_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive position management with entry, scaling, and exits
        
        Logic flow:
        1. Initialize position = 0
        2. Enter on signal
        3. Scale in if conditions improve (optional)
        4. Exit on any exit condition
        5. Track position size and pyramid level
        """
        df['Position'] = 0.0
        df['Position_Size'] = 0.0
        df['Pyramid_Level'] = 0
        
        current_position = 0
        pyramid_level = 0
        entry_zscore = None
        
        for i in range(1, len(df)):
            idx = df.index[i]
            prev_idx = df.index[i-1]
            
            # Carry forward previous position
            current_position = df.loc[prev_idx, 'Position']
            pyramid_level = df.loc[prev_idx, 'Pyramid_Level']
            
            # === Check Exit Conditions First ===
            if current_position != 0:
                # Mean reversion exit
                if df.loc[idx, 'Exit_Mean_Reversion']:
                    current_position = 0
                    pyramid_level = 0
                    entry_zscore = None
                
                # Stop-loss: z-score reverses strongly
                elif current_position > 0 and df.loc[idx, 'Z_Score'] > 0.5:
                    current_position = 0
                    pyramid_level = 0
                    entry_zscore = None
                
                elif current_position < 0 and df.loc[idx, 'Z_Score'] < -0.5:
                    current_position = 0
                    pyramid_level = 0
                    entry_zscore = None
                
                # Regime change exit
                elif df.loc[idx, 'Exit_Regime_Change']:
                    current_position = 0
                    pyramid_level = 0
                    entry_zscore = None
            
            # === Check Entry Conditions ===
            if current_position == 0:
                if df.loc[idx, 'Signal'] == 1:  # Long signal
                    current_position = 1.0
                    pyramid_level = 1
                    entry_zscore = df.loc[idx, 'Z_Score']
                
                elif df.loc[idx, 'Signal'] == -1:  # Short signal
                    current_position = -1.0
                    pyramid_level = 1
                    entry_zscore = df.loc[idx, 'Z_Score']
            
            # === Check Scaling Conditions ===
            elif (self.config.strategy.scale_in_enabled and 
                  pyramid_level < self.config.strategy.max_pyramid_levels):
                
                # Scale into long if z-score gets more negative
                if current_position > 0 and entry_zscore is not None:
                    if df.loc[idx, 'Z_Score'] < (entry_zscore - self.config.strategy.scale_in_threshold):
                        pyramid_level += 1
                        current_position = float(pyramid_level)
                        entry_zscore = df.loc[idx, 'Z_Score']
                
                # Scale into short if z-score gets more positive
                elif current_position < 0 and entry_zscore is not None:
                    if df.loc[idx, 'Z_Score'] > (entry_zscore + self.config.strategy.scale_in_threshold):
                        pyramid_level += 1
                        current_position = -float(pyramid_level)
                        entry_zscore = df.loc[idx, 'Z_Score']
            
            # Update dataframe
            df.loc[idx, 'Position'] = current_position
            df.loc[idx, 'Pyramid_Level'] = pyramid_level
        
        return df
    
    def _add_trade_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add useful metadata for trade analysis"""
        
        # Mark trade entry and exit points
        df['Trade_Entry'] = (df['Position'] != 0) & (df['Position'].shift(1) == 0)
        df['Trade_Exit'] = (df['Position'] == 0) & (df['Position'].shift(1) != 0)
        
        # Trade direction
        df['Trade_Direction'] = np.where(df['Position'] > 0, 'Long',
                                         np.where(df['Position'] < 0, 'Short', 'Flat'))
        
        # Days in trade
        df['Days_In_Trade'] = 0
        position_counter = 0
        
        for i in range(len(df)):
            if df['Position'].iloc[i] != 0:
                position_counter += 1
            else:
                position_counter = 0
            df['Days_In_Trade'].iloc[i] = position_counter
        
        return df
    
    def _print_signal_summary(self, df: pd.DataFrame):
        """Print summary statistics of generated signals"""
        
        total_days = len(df)
        entry_signals = df['Trade_Entry'].sum()
        long_entries = (df['Trade_Entry'] & (df['Position'] > 0)).sum()
        short_entries = (df['Trade_Entry'] & (df['Position'] < 0)).sum()
        
        days_in_position = (df['Position'] != 0).sum()
        pct_in_market = (days_in_position / total_days) * 100
        
        print("\n" + "=" * 60)
        print(f"📊 SIGNAL GENERATION SUMMARY: {self.pair_name}")
        print("=" * 60)
        print(f"Total trading days:     {total_days}")
        print(f"Total entry signals:    {entry_signals}")
        print(f"  - Long entries:       {long_entries}")
        print(f"  - Short entries:      {short_entries}")
        print(f"Days in position:       {days_in_position} ({pct_in_market:.1f}%)")
        
        if days_in_position > 0:
            avg_hold = df[df['Position'] != 0]['Days_In_Trade'].mean()
            print(f"Average hold time:      {avg_hold:.1f} days")
        
        # Regime breakdown
        if 'Regime_Composite' in df.columns:
            print(f"\nRegime distribution:")
            regime_counts = df['Regime_Composite'].value_counts()
            for regime, count in regime_counts.items():
                print(f"  - {regime}: {count} days ({count/total_days*100:.1f}%)")
        
        print("=" * 60)
    
    def get_trade_list(self) -> pd.DataFrame:
        """
        Extract list of individual trades for detailed analysis
        
        Returns:
            DataFrame with one row per trade
        """
        if self.signals is None:
            raise ValueError("Signals not generated. Call generate_signals() first.")
        
        trades = []
        in_trade = False
        entry_date = None
        entry_price = None
        entry_zscore = None
        direction = None
        
        for idx, row in self.signals.iterrows():
            # Trade entry
            if row['Trade_Entry']:
                in_trade = True
                entry_date = idx
                entry_price = row['Spread']
                entry_zscore = row['Z_Score']
                direction = 'Long' if row['Position'] > 0 else 'Short'
            
            # Trade exit
            elif row['Trade_Exit'] and in_trade:
                exit_date = idx
                exit_price = row['Spread']
                exit_zscore = row['Z_Score']
                
                # Calculate P&L
                if direction == 'Long':
                    pnl = exit_price - entry_price
                else:  # Short
                    pnl = entry_price - exit_price
                
                pnl_pct = (pnl / abs(entry_price)) * 100
                
                trades.append({
                    'Entry_Date': entry_date,
                    'Exit_Date': exit_date,
                    'Direction': direction,
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'Entry_Z_Score': entry_zscore,
                    'Exit_Z_Score': exit_zscore,
                    'Duration_Days': (exit_date - entry_date).days,
                    'PnL': pnl,
                    'PnL_Pct': pnl_pct,
                    'Win': pnl > 0
                })
                
                in_trade = False
        
        return pd.DataFrame(trades)
    
    def analyze_trades(self) -> Dict:
        """
        Comprehensive trade analysis
        
        Returns:
            Dictionary with trade statistics
        """
        trades_df = self.get_trade_list()
        
        if len(trades_df) == 0:
            return {"error": "No completed trades"}
        
        wins = trades_df[trades_df['Win']]
        losses = trades_df[~trades_df['Win']]
        
        analysis = {
            'total_trades': len(trades_df),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(trades_df) * 100,
            'avg_win': wins['PnL_Pct'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['PnL_Pct'].mean() if len(losses) > 0 else 0,
            'largest_win': trades_df['PnL_Pct'].max(),
            'largest_loss': trades_df['PnL_Pct'].min(),
            'avg_duration': trades_df['Duration_Days'].mean(),
            'long_trades': len(trades_df[trades_df['Direction'] == 'Long']),
            'short_trades': len(trades_df[trades_df['Direction'] == 'Short']),
        }
        
        # Expectancy (average $ per trade)
        analysis['expectancy'] = trades_df['PnL'].mean()
        
        # Profit factor
        gross_profit = wins['PnL'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['PnL'].sum()) if len(losses) > 0 else 1
        

        analysis['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        return analysis

