"""
Data acquisition and preprocessing for pairs trading strategies
Supports any two cointegrated assets (stocks, ETFs, futures, forex, etc.)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional, Dict, Literal
from statsmodels.tsa.stattools import adfuller, coint
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PairsDataHandler:
    """
    Universal data handler for pairs trading strategies
    
    Features:
    - Fetch and align any two financial instruments
    - Multiple spread calculation methods (log, simple, ratio)
    - Statistical validation (cointegration, stationarity, half-life)
    - Rolling cointegration analysis
    - Data quality checks and cleaning
    - Hedge ratio calculation
    
    Example:
        >>> handler = PairsDataHandler(config, asset1='SPY', asset2='QQQ')
        >>> df = handler.fetch_data()
        >>> handler.test_cointegration()
    """
    
    def __init__(self, config, asset1_ticker: str, asset2_ticker: str, 
                 pair_name: Optional[str] = None):
        """
        Initialize data handler for a specific pair
        
        Args:
            config: Configuration object
            asset1_ticker: First asset ticker (e.g., 'SPY', 'CL=F', 'EURUSD=X')
            asset2_ticker: Second asset ticker
            pair_name: Optional descriptive name (e.g., 'SPY-QQQ', 'Gold-Silver')
        """
        self.config = config
        self.asset1_ticker = asset1_ticker
        self.asset2_ticker = asset2_ticker
        self.pair_name = pair_name or f"{asset1_ticker}-{asset2_ticker}"
        
        self.asset1_data = None
        self.asset2_data = None
        self.df = None
        self.validation_results = {}
        
    def fetch_data(self, verbose: bool = True) -> pd.DataFrame:
        """
        Fetch historical data for both assets with validation
        
        Returns:
            pd.DataFrame: Cleaned and aligned price data
        """
        if verbose:
            print("=" * 60)
            print(f"📥 FETCHING DATA FOR {self.pair_name}")
            print("=" * 60)
        
        try:
            # Download Asset 1
            if verbose:
                print(f"Downloading {self.asset1_ticker} data...")
            asset1_raw = yf.download(
                self.asset1_ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=True
            )
            
            # Download Asset 2
            if verbose:
                print(f"Downloading {self.asset2_ticker} data...")
            asset2_raw = yf.download(
                self.asset2_ticker,
                start=self.config.start_date,
                end=self.config.end_date,
                progress=False,
                auto_adjust=True
            )
            
            # Clean data
            asset1_clean = self._clean_dataframe(asset1_raw)
            asset2_clean = self._clean_dataframe(asset2_raw)
            
            # Merge and align
            df = pd.DataFrame({
                'Asset1_Close': asset1_clean['Close'],
                'Asset1_High': asset1_clean['High'],
                'Asset1_Low': asset1_clean['Low'],
                'Asset1_Volume': asset1_clean['Volume'],
                'Asset2_Close': asset2_clean['Close'],
                'Asset2_High': asset2_clean['High'],
                'Asset2_Low': asset2_clean['Low'],
                'Asset2_Volume': asset2_clean['Volume']
            }).dropna()
            
            # Data quality checks
            df = self._quality_filter(df)
            
            # Store clean data
            self.df = df
            self.asset1_data = asset1_clean
            self.asset2_data = asset2_clean
            
            if verbose:
                print(f"\n✅ Successfully loaded {len(df)} trading days")
                print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
                print(f"   {self.asset1_ticker} range: ${df['Asset1_Close'].min():.2f} - ${df['Asset1_Close'].max():.2f}")
                print(f"   {self.asset2_ticker} range: ${df['Asset2_Close'].min():.2f} - ${df['Asset2_Close'].max():.2f}")
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {self.pair_name}: {str(e)}")
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean multi-index columns and handle missing data"""
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.droplevel(1)
        
        # Replace zeros with NaN
        df = df.replace(0, np.nan)
        
        # Forward fill small gaps (max 3 days)
        df = df.ffill(limit=3)
        
        # Drop remaining NaNs
        df = df.dropna()
        
        return df
    


    def _quality_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data quality filters"""
        original_len = len(df)
        
        # Remove prices below threshold
        df = df[
            (df['Asset1_Close'] > self.config.data.min_price_threshold) &  # ✅ FIXED
            (df['Asset2_Close'] > self.config.data.min_price_threshold)    # ✅ FIXED
        ]
        
        # Remove extreme outliers (z-score method)
        for col in ['Asset1_Close', 'Asset2_Close']:
            z_scores = np.abs(stats.zscore(df[col]))
            df = df[z_scores < self.config.data.outlier_std_threshold]  # ✅ FIXED
        
        # Remove days with zero volume (if volume exists)
        if 'Asset1_Volume' in df.columns and 'Asset2_Volume' in df.columns:
            df = df[(df['Asset1_Volume'] > 0) & (df['Asset2_Volume'] > 0)]
        
        removed = original_len - len(df)
        if removed > 0:
            print(f"   Filtered out {removed} low-quality data points ({removed/original_len*100:.2f}%)")
        
        return df

    def test_cointegration(self, verbose: bool = True) -> Dict[str, float]:
        """Test if the pair is cointegrated"""
        if verbose:
            print("\n" + "=" * 60)
            print(f"🔬 STATISTICAL VALIDATION: {self.pair_name}")
            print("=" * 60)
        
        asset1 = self.df['Asset1_Close'].values
        asset2 = self.df['Asset2_Close'].values
        
        # Engle-Granger cointegration test
        _, coint_pval, _ = coint(asset1, asset2)
        is_cointegrated = coint_pval < self.config.data.cointegration_pvalue  # ✅ FIXED
        
        if verbose:
            print(f"\n1. Cointegration Test (Engle-Granger)")
            print(f"   P-value: {coint_pval:.6f}")
            print(f"   Threshold: {self.config.data.cointegration_pvalue}")  # ✅ FIXED
            print(f"   Result: {'✅ COINTEGRATED' if is_cointegrated else '❌ NOT COINTEGRATED'}")
            print(f"   Interpretation: {'Pair exhibits mean-reversion' if is_cointegrated else 'Weak statistical relationship'}")
        
        results = {
            'coint_pvalue': coint_pval,
            'is_cointegrated': is_cointegrated
        }
        
        self.validation_results.update(results)
        return results







    def compute_spread(self, method: Literal['log', 'simple', 'ratio'] = 'log') -> pd.Series:
        """
        Compute spread between Asset1 and Asset2
        
        Args:
            method: 
                - 'log': log(Asset1) - log(Asset2) [default, best for mean-reversion]
                - 'simple': Asset1 - Asset2 [good for similar-priced assets]
                - 'ratio': Asset1 / Asset2 [good for percentage-based analysis]
        
        Returns:
            pd.Series: Computed spread
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        if method == 'log':
            spread = np.log(self.df['Asset1_Close']) - np.log(self.df['Asset2_Close'])
        elif method == 'simple':
            spread = self.df['Asset1_Close'] - self.df['Asset2_Close']
        elif method == 'ratio':
            spread = self.df['Asset1_Close'] / self.df['Asset2_Close']
        else:
            raise ValueError(f"Unknown method: {method}. Use 'log', 'simple', or 'ratio'")
        
        return spread
    
    
    def test_stationarity(self, spread: pd.Series, verbose: bool = True) -> Dict[str, float]:
        """
        Augmented Dickey-Fuller test for spread stationarity
        
        Args:
            spread: Spread series to test
            verbose: Print results
            
        Returns:
            Dict with stationarity test results
        """
        result = adfuller(spread.dropna(), autolag='AIC')
        adf_stat = result[0]
        pvalue = result[1]
        critical_values = result[4]
        is_stationary = pvalue < 0.05
        
        if verbose:
            print(f"\n2. Stationarity Test (Augmented Dickey-Fuller)")
            print(f"   ADF Statistic: {adf_stat:.4f}")
            print(f"   P-value: {pvalue:.6f}")
            print(f"   Critical values: 1%={critical_values['1%']:.3f}, "
                  f"5%={critical_values['5%']:.3f}, 10%={critical_values['10%']:.3f}")
            print(f"   Result: {'✅ STATIONARY' if is_stationary else '❌ NON-STATIONARY'}")
        
        results = {
            'adf_statistic': adf_stat,
            'adf_pvalue': pvalue,
            'is_stationary': is_stationary
        }
        
        self.validation_results.update(results)
        return results
    
    def calculate_half_life(self, spread: pd.Series, verbose: bool = True) -> float:
        """
        Calculate mean-reversion speed (half-life) using Ornstein-Uhlenbeck
        
        Returns:
            float: Half-life in days (lower = faster mean-reversion)
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align series
        spread_lag, spread_diff = spread_lag.align(spread_diff, join='inner')
        
        # OLS regression: Δy_t = α + β*y_{t-1} + ε
        beta = np.polyfit(spread_lag, spread_diff, 1)[0]
        half_life = -np.log(2) / beta if beta < 0 else np.inf
        
        if verbose:
            print(f"\n3. Mean-Reversion Speed (Half-Life)")
            print(f"   Half-life: {half_life:.2f} days")
            if half_life < 30:
                print(f"   Assessment: ✅ FAST mean-reversion (excellent)")
            elif half_life < 60:
                print(f"   Assessment: ✅ MODERATE mean-reversion (good)")
            else:
                print(f"   Assessment: ⚠️  SLOW mean-reversion (patience required)")
        
        self.validation_results['half_life'] = half_life
        return half_life
    
    def calculate_hedge_ratio(self, method: Literal['ols', 'tls'] = 'ols') -> float:
        """
        Calculate optimal hedge ratio (beta) between assets
        
        Args:
            method: 'ols' for ordinary least squares, 'tls' for total least squares
        
        Returns:
            float: Hedge ratio (units of Asset2 per unit of Asset1)
        """
        asset1 = self.df['Asset1_Close'].values
        asset2 = self.df['Asset2_Close'].values
        
        if method == 'ols':
            # OLS: Asset1 = α + β*Asset2 + ε
            beta = np.polyfit(asset2, asset1, 1)[0]
        elif method == 'tls':
            # Total Least Squares (accounts for noise in both variables)
            from scipy.linalg import svd
            X = np.vstack([asset2, asset1]).T
            X_centered = X - X.mean(axis=0)
            U, s, Vt = svd(X_centered)
            beta = Vt[0, 1] / Vt[0, 0]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"\n4. Hedge Ratio (Beta)")
        print(f"   β ({self.asset1_ticker}/{self.asset2_ticker}): {beta:.4f}")
        print(f"   Interpretation: {beta:.4f} units of {self.asset2_ticker} hedge 1 unit of {self.asset1_ticker}")
        
        self.validation_results['hedge_ratio'] = beta
        return beta
    
    def calculate_rolling_cointegration(self, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling cointegration to identify regime changes
        
        Args:
            window: Rolling window in days (252=1yr, 126=6mo)
        
        Returns:
            DataFrame with rolling p-values and cointegration status
        """
        print(f"\n🔄 Calculating rolling cointegration (window={window} days)...")
        
        asset1 = self.df['Asset1_Close'].values
        asset2 = self.df['Asset2_Close'].values
        
        rolling_results = []
        
        for i in range(window, len(asset1)):
            asset1_window = asset1[i-window:i]
            asset2_window = asset2[i-window:i]
            
            try:
                _, pvalue, _ = coint(asset1_window, asset2_window)
                rolling_results.append({
                    'Date': self.df.index[i],
                    'Coint_PValue': pvalue,
                    'Is_Cointegrated': pvalue < 0.05
                })
            except Exception:
                rolling_results.append({
                    'Date': self.df.index[i],
                    'Coint_PValue': np.nan,
                    'Is_Cointegrated': False
                })
        
        results_df = pd.DataFrame(rolling_results).set_index('Date')
        
        # Summary stats
        valid_pvals = results_df['Coint_PValue'].dropna()
        pct_cointegrated = (results_df['Is_Cointegrated'].sum() / len(results_df)) * 100
        
        print(f"✅ Analysis complete:")
        print(f"   Cointegrated: {pct_cointegrated:.1f}% of the time")
        print(f"   Mean p-value: {valid_pvals.mean():.4f}")
        
        return results_df
    
    def prepare_strategy_data(self, spread_method: str = 'log') -> pd.DataFrame:
        """
        One-stop method to prepare complete dataset with all features
        
        Args:
            spread_method: Method for spread calculation
        
        Returns:
            DataFrame ready for strategy backtesting
        """
        # Fetch data if not loaded
        if self.df is None:
            self.fetch_data()
        
        df = self.df.copy()
        
        # Calculate spread
        df['Spread'] = self.compute_spread(method=spread_method)
        
        # Calculate hedge ratio
        hedge_ratio = self.calculate_hedge_ratio(method='ols')
        df['Hedge_Ratio'] = hedge_ratio
        
        # Run validation tests
        self.test_cointegration()
        self.test_stationarity(df['Spread'])
        self.calculate_half_life(df['Spread'])
        
        # Print summary
        print(self.generate_summary_report())
        
        return df
    
    def generate_summary_report(self) -> str:
        """Generate summary of all validation tests"""
        if not self.validation_results:
            return "No validation performed yet."
        
        report = "\n" + "=" * 60
        report += f"\n📊 VALIDATION SUMMARY: {self.pair_name}"
        report += "\n" + "=" * 60
        
        # Cointegration
        if self.validation_results.get('is_cointegrated'):
            report += "\n✅ Pair is cointegrated - suitable for mean-reversion"
        else:
            report += "\n❌ Pair NOT cointegrated - high risk"
        
        # Stationarity
        if self.validation_results.get('is_stationary'):
            report += "\n✅ Spread is stationary - predictable behavior"
        else:
            report += "\n❌ Spread is non-stationary - may trend"
        
        # Half-life
        hl = self.validation_results.get('half_life', float('inf'))
        if hl < 60:
            report += f"\n✅ Half-life of {hl:.1f} days - good for trading"
        else:
            report += f"\n⚠️  Half-life of {hl:.1f} days - slow mean-reversion"
        
        # Overall
        report += "\n" + "-" * 60
        all_good = (
            self.validation_results.get('is_cointegrated', False) and
            self.validation_results.get('is_stationary', False) and
            hl < 60
        )
        
        if all_good:
            report += f"\n🎯 VERDICT: EXCELLENT pair for mean-reversion"
        else:
            report += f"\n⚠️  VERDICT: Proceed with caution"
        
        report += "\n" + "=" * 60 + "\n"
        return report
    
    def get_pair_info(self) -> Dict:
        """Return information about the current pair"""
        return {
            'pair_name': self.pair_name,
            'asset1': self.asset1_ticker,
            'asset2': self.asset2_ticker,
            'data_points': len(self.df) if self.df is not None else 0,
            'date_range': (
                f"{self.df.index[0].date()} to {self.df.index[-1].date()}"
                if self.df is not None else "Not loaded"
            ),
            'validation': self.validation_results
        }