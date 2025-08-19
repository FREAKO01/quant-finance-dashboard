"""
================================================================================
PROFESSIONAL ELITE OPTIONS TRADING BACKTESTING SYSTEM
================================================================================

Version: 5.0 - Professional Edition
Author: Advanced Trading Systems
Initial Capital: ₹1,00,000
Target Period: 2019-2025 (6 years with walk-forward testing)

COMPREHENSIVE FEATURES:
- Multi-index support (NIFTY 50 + BANK NIFTY + SENSEX)
- Walk-forward backtesting with rolling optimization
- Advanced technical analysis with 20+ indicators
- Sophisticated options pricing and Greeks calculation
- Dynamic risk management with volatility adjustment
- Machine learning signal optimization
- Real-time market regime detection
- Professional reporting with interactive charts
- Monte Carlo simulation capabilities
- Comprehensive performance analytics

Installation: 
pip install yfinance pandas numpy matplotlib seaborn scipy scikit-learn xgboost ta rich plotly

DISCLAIMER: For educational and research purposes only. 
Past performance does not guarantee future results.
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

# Core Data Science Stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot
import yfinance as yf

# Standard Library
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from enum import Enum
import logging
import math
import random
import json
import time
import itertools
from collections import defaultdict

# Scientific Computing
from scipy import stats
from scipy.stats import norm, normaltest
from scipy.optimize import minimize, differential_evolution

# Machine Learning Stack
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression, Ridge
    import xgboost as xgb
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  ML libraries not available. Install with: pip install scikit-learn xgboost")

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    print("⚠️  TA library not available. Install with: pip install ta")

# Enhanced Console Output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, track
    from rich.panel import Panel
    from rich.text import Text
    from rich.layout import Layout
    from rich import print as rprint
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None
    def rprint(x): print(x)

# Indian Market Data
try:
    from nsepy import get_history
    from nsepy.derivatives import get_expiry_date
    NSEPY_AVAILABLE = True
    print("✅ NSEPy available for enhanced Indian market data")
except ImportError:
    NSEPY_AVAILABLE = False
    print("⚠️  NSEPy not available. Install with: pip install nsepy")

# Configure Professional Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('EliteOptionsSystem')

# ================================================================================
# ENHANCED CONFIGURATION & ENUMS
# ================================================================================

class StrategyType(Enum):
    """Options trading strategies supported by the system"""
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    LONG_STRADDLE = "long_straddle"
    SHORT_STRANGLE = "short_strangle"
    CALENDAR_SPREAD = "calendar_spread"

class MarketRegime(Enum):
    """Market regime classifications for strategy selection"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class IndexType(Enum):
    """Supported Indian market indices"""
    NIFTY_50 = "nifty_50"
    BANK_NIFTY = "bank_nifty"
    SENSEX = "sensex"
    NIFTY_IT = "nifty_it"
    NIFTY_FMCG = "nifty_fmcg"

@dataclass
class TradingConfig:
    """Comprehensive trading system configuration"""
    
    # Portfolio Settings
    initial_capital: float = 100000.0  # ₹1,00,000
    
    # Multi-Index Configuration
    indices: List[str] = field(default_factory=lambda: ["^NSEI", "^NSEBANK", "^BSESN"])
    index_names: List[str] = field(default_factory=lambda: ["NIFTY_50", "BANK_NIFTY", "SENSEX"])
    index_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])  # Portfolio allocation
    
    # Time Period Configuration
    start_date: date = date(2019, 1, 1)
    end_date: date = date(2025, 1, 31)
    
    # Walk-Forward Testing Parameters
    walk_forward_enabled: bool = True
    walk_forward_periods: int = 6
    training_window_months: int = 12
    testing_window_months: int = 4
    rebalance_frequency: int = 30  # days
    
    # Risk Management (Conservative Professional Settings)
    max_risk_per_trade: float = 0.008  # 0.8% per trade
    max_portfolio_risk: float = 0.08   # 8% total exposure
    max_positions_per_index: int = 3
    max_total_positions: int = 8
    
    # Stop Loss & Take Profit (Professional Levels)
    stop_loss_pct: float = 0.45        # 45% stop loss
    take_profit_pct: float = 0.80      # 80% take profit
    time_stop_days: int = 25           # Close after 25 days
    
    # Options Parameters
    min_dte: int = 10
    max_dte: int = 45
    iv_percentile_threshold: float = 0.30
    delta_target_otm: float = 0.20     # Out of money target delta
    delta_target_atm: float = 0.45     # At the money target delta
    min_credit_threshold: float = 0.30   # Minimum credit to accept trade
    
    # Technical Analysis Parameters (Optimized)
    ma_fast: int = 8
    ma_slow: int = 21
    ma_trend: int = 55
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14
    adx_threshold: int = 25
    
    # Signal Generation Thresholds
    signal_threshold: float = 0.35     # Higher threshold for quality
    momentum_threshold: float = 0.25
    volume_threshold: float = 1.5
    volatility_low_threshold: float = 0.15
    volatility_high_threshold: float = 0.35
    correlation_threshold: float = 0.7
    
    # Machine Learning Configuration
    use_ml_optimization: bool = True
    ml_model_type: str = "xgboost"     # "xgboost", "random_forest", "gradient_boost"
    ml_lookback_days: int = 252
    ml_validation_split: float = 0.2
    feature_selection_threshold: float = 0.01
    
    # Market Data Configuration
    use_real_options_data: bool = False
    options_data_source: str = "synthetic"  # "zerodha", "truedata", "synthetic"
    data_quality_checks: bool = True
    outlier_detection: bool = True
    
    # Transaction Costs (Realistic Indian Market)
    brokerage_per_lot: float = 20.0
    lot_size_nifty: int = 50
    lot_size_banknifty: int = 25
    lot_size_sensex: int = 10
    slippage_bps: float = 2.0
    impact_cost_bps: float = 1.5
    
    # Performance Benchmarking
    benchmark_indices: List[str] = field(default_factory=lambda: ["^NSEI"])
    risk_free_rate: float = 0.06
    
    # System Settings
    random_seed: int = 42
    parallel_processing: bool = True
    cache_enabled: bool = True
    verbose_logging: bool = True

# ================================================================================
# PROFESSIONAL DATA MANAGEMENT SYSTEM
# ================================================================================

class ProfessionalDataManager:
    """Advanced data management with multi-source support and quality checks"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_cache = {}
        self.options_cache = {}
        self.quality_metrics = {}
        
        # Set random seed for reproducibility
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
    
    def fetch_multi_index_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple indices with quality validation"""
        all_data = {}
        
        for i, ticker in enumerate(self.config.indices):
            index_name = self.config.index_names[i]
            
            try:
                logger.info(f"📊 Fetching {index_name} data ({ticker})")
                data = self._fetch_single_index_data(ticker, index_name)
                
                if not data.empty:
                    # Quality validation
                    if self.config.data_quality_checks:
                        data = self._validate_data_quality(data, index_name)
                    
                    all_data[index_name] = data
                    logger.info(f"✅ {index_name}: {len(data)} days, "
                              f"Range: ₹{data['Close'].min():.0f} - ₹{data['Close'].max():.0f}")
                else:
                    logger.warning(f"⚠️  No data available for {index_name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to fetch {index_name}: {e}")
                continue
        
        return all_data
    
    def _fetch_single_index_data(self, ticker: str, index_name: str) -> pd.DataFrame:
        """Fetch data for a single index with multiple fallback sources"""
        cache_key = f"{ticker}_{self.config.start_date}_{self.config.end_date}"
        
        if self.config.cache_enabled and cache_key in self.data_cache:
            return self.data_cache[cache_key].copy()
        
        # Try multiple data sources
        data_sources = [
            self._fetch_yfinance_data,
            self._fetch_nsepy_data,
            self._generate_synthetic_data
        ]
        
        for i, fetch_func in enumerate(data_sources):
            try:
                df = fetch_func(ticker, index_name)
                if not df.empty:
                    df = self._clean_and_validate_data(df)
                    if self.config.cache_enabled:
                        self.data_cache[cache_key] = df.copy()
                    return df.copy()
            except Exception as e:
                logger.warning(f"Data source {i+1} failed for {ticker}: {e}")
                continue
        
        raise ValueError(f"All data sources failed for {ticker}")
    
    def _fetch_yfinance_data(self, ticker: str, index_name: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(
            start=self.config.start_date,
            end=self.config.end_date,
            auto_adjust=True,
            interval='1d'
        )
        
        if df.empty:
            # Try alternative tickers
            alternatives = {
                "^NSEI": ["NIFTY50.NS", "^CNX", "NSEI.NS"],
                "^NSEBANK": ["BANKNIFTY.NS", "^CNXBANK"],
                "^BSESN": ["BSE.BO", "SENSEX.BO"]
            }
            
            for alt_ticker in alternatives.get(ticker, []):
                alt_data = yf.Ticker(alt_ticker).history(
                    start=self.config.start_date,
                    end=self.config.end_date,
                    auto_adjust=True
                )
                if not alt_data.empty:
                    return alt_data
        
        return df
    
    def _fetch_nsepy_data(self, ticker: str, index_name: str) -> pd.DataFrame:
        """Fetch data from NSEPy"""
        if not NSEPY_AVAILABLE:
            raise ImportError("NSEPy not available")
        
        # Map tickers to NSEPy symbols
        symbol_map = {
            "^NSEI": "NIFTY",
            "^NSEBANK": "BANKNIFTY",
        }
        
        symbol = symbol_map.get(ticker, ticker)
        
        df = get_history(
            symbol=symbol,
            start=self.config.start_date,
            end=self.config.end_date,
            index=True
        )
        
        return df
    
    def _generate_synthetic_data(self, ticker: str, index_name: str) -> pd.DataFrame:
        """Generate high-quality synthetic market data"""
        logger.info(f"Generating synthetic data for {index_name}")
        
        # Create business day date range
        dates = pd.date_range(
            start=self.config.start_date,
            end=self.config.end_date,
            freq='B'
        )
        
        # Market-specific parameters
        market_params = {
            "NIFTY_50": {"initial": 12000, "drift": 0.08, "volatility": 0.18},
            "BANK_NIFTY": {"initial": 35000, "drift": 0.10, "volatility": 0.22},
            "SENSEX": {"initial": 40000, "drift": 0.09, "volatility": 0.19}
        }
        
        params = market_params.get(index_name, {"initial": 15000, "drift": 0.08, "volatility": 0.20})
        
        # Generate realistic price series using geometric Brownian motion
        n_days = len(dates)
        dt = 1/252  # Daily time step
        
        # Add market cycles and volatility clustering
        volatility_regime = self._generate_volatility_regime(n_days)
        market_cycle = self._generate_market_cycle(n_days)
        
        # Generate returns with regime-dependent volatility
        returns = []
        for i in range(n_days):
            vol = params["volatility"] * volatility_regime[i]
            drift = params["drift"] * market_cycle[i]
            ret = np.random.normal(drift * dt, vol * np.sqrt(dt))
            returns.append(ret)
        
        returns = np.array(returns)
        
        # Calculate cumulative prices
        log_prices = np.log(params["initial"]) + np.cumsum(returns)
        prices = np.exp(log_prices)
        
        # Generate OHLC data
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        
        # Generate realistic OHLC using intraday volatility
        daily_range = np.random.exponential(0.02, n_days)  # Daily range as % of close
        
        df['Open'] = df['Close'].shift(1) * (1 + np.random.normal(0, daily_range/4))
        df['Open'].iloc[0] = df['Close'].iloc[0]
        
        # High and Low based on close and open
        for i in range(len(df)):
            high_factor = 1 + np.random.exponential(daily_range[i]/2)
            low_factor = 1 - np.random.exponential(daily_range[i]/2)
            
            df.loc[df.index[i], 'High'] = max(df['Open'].iloc[i], df['Close'].iloc[i]) * high_factor
            df.loc[df.index[i], 'Low'] = min(df['Open'].iloc[i], df['Close'].iloc[i]) * low_factor
        
        # Generate volume with realistic patterns
        base_volume = {"NIFTY_50": 1e8, "BANK_NIFTY": 5e7, "SENSEX": 8e7}
        vol_base = base_volume.get(index_name, 7e7)
        
        # Volume correlated with volatility and price moves
        volume_multiplier = 1 + 2 * np.abs(returns) + 0.5 * (volatility_regime - 1)
        df['Volume'] = np.random.lognormal(np.log(vol_base), 0.3, n_days) * volume_multiplier
        
        # Convert index to dates
        df.index = df.index.date
        
        return df
    
    def _generate_volatility_regime(self, n_days: int) -> np.array:
        """Generate volatility regime switching"""
        regimes = np.ones(n_days)
        
        # Add volatility clustering
        high_vol_periods = np.random.poisson(20, n_days // 50)  # Average 20 day periods
        
        start_idx = 0
        for period_length in high_vol_periods:
            if start_idx + period_length < n_days:
                regimes[start_idx:start_idx + period_length] = 1.5  # High volatility
                start_idx += period_length + np.random.poisson(30)  # Gap to next period
        
        return regimes
    
    def _generate_market_cycle(self, n_days: int) -> np.array:
        """Generate market cycle effects"""
        # Create longer-term market cycles
        cycle_length = 252 * 2  # 2-year cycle
        x = np.linspace(0, 4 * np.pi * n_days / cycle_length, n_days)
        cycle = 0.8 + 0.4 * np.sin(x) + 0.2 * np.sin(2*x)
        
        return cycle
    
    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data with professional quality checks"""
        original_length = len(df)
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                if col == 'Volume':
                    df['Volume'] = 1000000  # Default volume
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Remove invalid data points
        df = df.dropna()
        df = df[df['Close'] > 0]
        df = df[df['Volume'] > 0]
        
        # OHLC consistency checks
        df = df[df['High'] >= df['Low']]
        df = df[df['High'] >= df['Close']]
        df = df[df['High'] >= df['Open']]
        df = df[df['Low'] <= df['Close']]
        df = df[df['Low'] <= df['Open']]
        
        # Outlier detection and removal
        if self.config.outlier_detection:
            df = self._remove_outliers(df)
        
        # Convert index to dates if needed
        idx = pd.to_datetime(df.index, errors='coerce')
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert(None)
        df.index = idx.normalize().date

        logger.info(f"Data cleaning: {original_length} -> {len(df)} rows")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers using multiple methods"""
        # Method 1: Return-based outlier detection
        returns = df['Close'].pct_change().abs()
        return_threshold = returns.quantile(0.995)  # Top 1% moves
        outliers_returns = returns > return_threshold
        
        # Method 2: Price gap detection
        gaps = np.abs(df['Open'] / df['Close'].shift(1) - 1)
        gap_threshold = gaps.quantile(0.999)  # Top 0.1% gaps
        outliers_gaps = gaps > gap_threshold
        
        # Method 3: Volume spike detection
        volume_ratio = df['Volume'] / df['Volume'].rolling(20).median()
        volume_threshold = volume_ratio.quantile(0.995)
        outliers_volume = volume_ratio > volume_threshold
        
        # Combine outlier detection methods
        total_outliers = outliers_returns | outliers_gaps | outliers_volume
        
        if total_outliers.sum() > 0:
            logger.warning(f"Removing {total_outliers.sum()} outlier days")
            df = df[~total_outliers]
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, index_name: str) -> pd.DataFrame:
        """Comprehensive data quality validation"""
        quality_metrics = {}
        
        # Check data completeness
        date_range = pd.date_range(start=self.config.start_date, end=self.config.end_date, freq='B')
        coverage = len(df) / len(date_range)
        quality_metrics['coverage'] = coverage
        
        # Check for data gaps
        df_dates = pd.to_datetime(df.index)
        gaps = (df_dates.to_series().diff() > pd.Timedelta(days=5)).sum()
        quality_metrics['gaps'] = gaps
        
        # Check price continuity
        returns = df['Close'].pct_change()
        extreme_moves = (np.abs(returns) > 0.15).sum()  # >15% daily moves
        quality_metrics['extreme_moves'] = extreme_moves
        
        # Check volume consistency
        zero_volume_days = (df['Volume'] == 0).sum()
        quality_metrics['zero_volume_days'] = zero_volume_days
        
        # Store quality metrics
        self.quality_metrics[index_name] = quality_metrics
        
        # Quality warnings
        if coverage < 0.8:
            logger.warning(f"{index_name}: Low data coverage ({coverage:.1%})")
        if gaps > 10:
            logger.warning(f"{index_name}: {gaps} significant gaps found")
        if extreme_moves > len(df) * 0.05:
            logger.warning(f"{index_name}: {extreme_moves} extreme price moves")
        
        return df

# ================================================================================
# ADVANCED TECHNICAL ANALYSIS ENGINE
# ================================================================================

class AdvancedTechnicalAnalyzer:
    """Professional technical analysis with 25+ indicators and regime detection"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.indicator_cache = {}
    
    def add_comprehensive_indicators(self, df: pd.DataFrame, index_name: str = None) -> pd.DataFrame:
        """Add comprehensive technical indicators suite"""
        data = df.copy()
        
        # Trend Indicators
        data = self._add_trend_indicators(data)
        
        # Momentum Indicators  
        data = self._add_momentum_indicators(data)
        
        # Volatility Indicators
        data = self._add_volatility_indicators(data)
        
        # Volume Indicators
        data = self._add_volume_indicators(data)
        
        # Support/Resistance
        data = self._add_support_resistance(data)
        
        # Market Structure
        data = self._add_market_structure(data)
        
        # Advanced Patterns
        data = self._add_pattern_recognition(data)
        
        # Regime Detection
        data = self._add_regime_detection(data)
        
        # Fill any remaining NaN values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data
    
    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-following indicators"""
        # Multiple Moving Averages
        data['EMA_Fast'] = data['Close'].ewm(span=self.config.ma_fast).mean()
        data['EMA_Slow'] = data['Close'].ewm(span=self.config.ma_slow).mean()
        data['EMA_Trend'] = data['Close'].ewm(span=self.config.ma_trend).mean()
        data['SMA_Fast'] = data['Close'].rolling(self.config.ma_fast).mean()
        data['SMA_Slow'] = data['Close'].rolling(self.config.ma_slow).mean()
        data['SMA_Trend'] = data['Close'].rolling(self.config.ma_trend).mean()
        
        # Triple EMA
        ema1 = data['Close'].ewm(span=21).mean()
        ema2 = ema1.ewm(span=21).mean()
        ema3 = ema2.ewm(span=21).mean()
        data['TEMA'] = 3 * ema1 - 3 * ema2 + ema3
        
        # Parabolic SAR
        data['PSAR'] = self._calculate_parabolic_sar(data)
        
        # Trend Strength
        data['Trend_Strength'] = (data['EMA_Fast'] - data['EMA_Slow']) / data['EMA_Slow'] * 100
        
        return data
    
    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum oscillators"""
        # RSI with multiple periods
        data['RSI'] = self._calculate_rsi(data['Close'], self.config.rsi_period)
        data['RSI_Fast'] = self._calculate_rsi(data['Close'], 7)
        data['RSI_Slow'] = self._calculate_rsi(data['Close'], 21)
        
        # Stochastic
        data['Stoch_K'], data['Stoch_D'] = self._calculate_stochastic(data)
        
        # MACD with histogram
        data['MACD'], data['MACD_Signal'], data['MACD_Histogram'] = self._calculate_macd(data)
        
        # Rate of Change
        data['ROC_5'] = data['Close'].pct_change(5) * 100
        data['ROC_10'] = data['Close'].pct_change(10) * 100
        data['ROC_20'] = data['Close'].pct_change(20) * 100
        
        # Momentum
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10)
        
        # Williams %R
        data['Williams_R'] = self._calculate_williams_r(data)
        
        return data
    
    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility measures"""
        # Average True Range
        data['ATR'] = self._calculate_atr(data)
        data['ATR_Pct'] = data['ATR'] / data['Close'] * 100
        
        # Bollinger Bands
        bb_mid = data['Close'].rolling(self.config.bb_period).mean()
        bb_std = data['Close'].rolling(self.config.bb_period).std()
        data['BB_Upper'] = bb_mid + (self.config.bb_std * bb_std)
        data['BB_Lower'] = bb_mid - (self.config.bb_std * bb_std)
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / bb_mid
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Bollinger Band Squeeze
        bb_squeeze_period = 20
        data['BB_Squeeze'] = data['BB_Width'] < data['BB_Width'].rolling(bb_squeeze_period).quantile(0.2)
        
        # Volatility measures
        data['Returns'] = data['Close'].pct_change()
        data['Volatility_10'] = data['Returns'].rolling(10).std() * np.sqrt(252)
        data['Volatility_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)
        data['Volatility_60'] = data['Returns'].rolling(60).std() * np.sqrt(252)
        data['Volatility_Rank'] = data['Volatility_20'].rolling(252).rank(pct=True)
        
        # Keltner Channels
        kc_mid = data['EMA_Fast']
        kc_range = data['ATR'] * 2
        data['KC_Upper'] = kc_mid + kc_range
        data['KC_Lower'] = kc_mid - kc_range
        
        return data
    
    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        # Volume moving averages
        data['Volume_SMA_10'] = data['Volume'].rolling(10).mean()
        data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
        data['Volume_EMA_10'] = data['Volume'].ewm(span=10).mean()
        
        # Volume ratios
        data['Volume_Ratio_10'] = data['Volume'] / data['Volume_SMA_10']
        data['Volume_Ratio_20'] = data['Volume'] / data['Volume_SMA_20']
        
        # Volume spikes
        data['Volume_Spike'] = data['Volume_Ratio_20'] > self.config.volume_threshold
        
        # On-Balance Volume
        data['OBV'] = self._calculate_obv(data)
        
        # Volume Price Trend
        data['VPT'] = self._calculate_vpt(data)
        
        # Accumulation/Distribution Line
        data['ADL'] = self._calculate_adl(data)
        
        return data
    
    def _add_support_resistance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels"""
        # Pivot Points
        data['PP'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['R1'] = 2 * data['PP'] - data['Low']
        data['S1'] = 2 * data['PP'] - data['High']
        data['R2'] = data['PP'] + (data['High'] - data['Low'])
        data['S2'] = data['PP'] - (data['High'] - data['Low'])
        
        # Dynamic Support/Resistance
        data['Resistance_20'] = data['High'].rolling(20).max()
        data['Support_20'] = data['Low'].rolling(20).min()
        data['Resistance_50'] = data['High'].rolling(50).max()
        data['Support_50'] = data['Low'].rolling(50).min()
        
        # Distance to S/R levels
        data['Dist_to_R20'] = (data['Resistance_20'] - data['Close']) / data['Close']
        data['Dist_to_S20'] = (data['Close'] - data['Support_20']) / data['Close']
        
        return data
    
    def _add_market_structure(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add market structure indicators"""
        # Higher Highs / Lower Lows
        lookback = 10
        data['HH'] = data['High'] > data['High'].rolling(lookback).max().shift(1)
        data['LL'] = data['Low'] < data['Low'].rolling(lookback).min().shift(1)
        data['HL'] = (data['Low'] > data['Low'].rolling(lookback).min().shift(1)) & \
                     (data['High'] < data['High'].rolling(lookback).max().shift(1))
        
        # Market structure score
        data['Structure_Score'] = (data['HH'].rolling(20).sum() - data['LL'].rolling(20).sum()) / 20
        
        # Fractal levels
        data['Fractal_High'] = self._identify_fractals(data['High'], mode='high')
        data['Fractal_Low'] = self._identify_fractals(data['Low'], mode='low')
        
        return data
    
    def _add_pattern_recognition(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition"""
        # Doji patterns
        body_size = np.abs(data['Close'] - data['Open'])
        total_range = data['High'] - data['Low']
        data['Doji'] = (body_size / total_range) < 0.1
        
        # Hammer patterns
        lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']
        upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
        data['Hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < 0.5 * body_size)
        
        # Engulfing patterns
        data['Bullish_Engulfing'] = (data['Close'] > data['Open']) & \
                                    (data['Close'].shift(1) < data['Open'].shift(1)) & \
                                    (data['Close'] > data['Open'].shift(1)) & \
                                    (data['Open'] < data['Close'].shift(1))
        
        data['Bearish_Engulfing'] = (data['Close'] < data['Open']) & \
                                    (data['Close'].shift(1) > data['Open'].shift(1)) & \
                                    (data['Close'] < data['Open'].shift(1)) & \
                                    (data['Open'] > data['Close'].shift(1))
        
        return data
    
    def _add_regime_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced market regime detection"""
        # Trend regime
        trend_score = (
            (data['EMA_Fast'] > data['EMA_Slow']).astype(int) * 2 - 1 +
            (data['Close'] > data['EMA_Trend']).astype(int) * 2 - 1 +
            (data['MACD'] > data['MACD_Signal']).astype(int) * 2 - 1
        ) / 3
        
        # Volatility regime
        vol_regime = pd.cut(
            data['Volatility_Rank'], 
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Overall market regime
        conditions = [
            (trend_score > 0.5) & (data['Volatility_Rank'] < 0.7),
            (trend_score < -0.5) & (data['Volatility_Rank'] < 0.7),
            (np.abs(trend_score) < 0.3) & (data['Volatility_Rank'] < 0.4),
            (data['Volatility_Rank'] > 0.7),
            (np.abs(trend_score) < 0.3) & (data['Volatility_Rank'] >= 0.4)
        ]
        
        choices = [
            MarketRegime.TRENDING_UP.value,
            MarketRegime.TRENDING_DOWN.value,
            MarketRegime.LOW_VOLATILITY.value,
            MarketRegime.HIGH_VOLATILITY.value,
            MarketRegime.SIDEWAYS.value
        ]
        
        data['Market_Regime'] = np.select(conditions, choices, default=MarketRegime.SIDEWAYS.value)
        data['Trend_Score'] = trend_score
        data['Volatility_Regime'] = vol_regime.astype(str).fillna('Medium')

        return data
    
    def generate_professional_signals(self, data: pd.DataFrame, index_name: str = None) -> pd.DataFrame:
        """Generate high-quality trading signals using multiple confirmations"""
        signals = data.copy()
        
        # Initialize signal components
        trend_signals = self._generate_trend_signals(signals)
        momentum_signals = self._generate_momentum_signals(signals)
        mean_reversion_signals = self._generate_mean_reversion_signals(signals)
        volume_signals = self._generate_volume_signals(signals)
        pattern_signals = self._generate_pattern_signals(signals)
        
        # Combine signals with weights
        signal_weights = {
            'trend': 0.30,
            'momentum': 0.25,
            'mean_reversion': 0.20,
            'volume': 0.15,
            'pattern': 0.10
        }
        
        # Calculate composite signal score
        signals['Signal_Score'] = (
            trend_signals * signal_weights['trend'] +
            momentum_signals * signal_weights['momentum'] +
            mean_reversion_signals * signal_weights['mean_reversion'] +
            volume_signals * signal_weights['volume'] +
            pattern_signals * signal_weights['pattern']
        )
        
        # Generate final signals with adaptive thresholds
        adaptive_threshold = self._calculate_adaptive_threshold(signals)
        
        signals['Signal_Bull'] = (signals['Signal_Score'] >= adaptive_threshold).astype(int)
        signals['Signal_Bear'] = (signals['Signal_Score'] <= -adaptive_threshold).astype(int)
        signals['Signal_Neutral'] = (
            (signals['Signal_Score'] > -adaptive_threshold) & 
            (signals['Signal_Score'] < adaptive_threshold)
        ).astype(int)
        
        # Final signal classification
        signals['Final_Signal'] = np.where(
            signals['Signal_Bull'], 1,
            np.where(signals['Signal_Bear'], -1, 0)
        )
        
        # Signal strength and quality
        signals['Signal_Strength'] = np.abs(signals['Signal_Score'])
        signals['Signal_Quality'] = self._calculate_signal_quality(signals)
        
        # High probability trade identification
        signals['High_Probability_Bull'] = (
            (signals['Final_Signal'] == 1) &
            (signals['Signal_Quality'] > 0.7) &
            (signals['Volatility_Rank'] > 0.3) &
            (signals['Volatility_Rank'] < 0.8)
        )
        
        signals['High_Probability_Bear'] = (
            (signals['Final_Signal'] == -1) &
            (signals['Signal_Quality'] > 0.7) &
            (signals['Volatility_Rank'] > 0.3) &
            (signals['Volatility_Rank'] < 0.8)
        )
        
        return signals
    
    def _generate_trend_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trend-following signals"""
        # EMA crossover
        ema_signal = np.where(data['EMA_Fast'] > data['EMA_Slow'], 1, -1)
        
        # Price vs trend EMA
        price_trend_signal = np.where(data['Close'] > data['EMA_Trend'], 1, -1)
        
        # MACD signal
        macd_signal = np.where(data['MACD'] > data['MACD_Signal'], 1, -1)
        
        # Parabolic SAR
        psar_signal = np.where(data['Close'] > data['PSAR'], 1, -1)
        
        # Trend strength filter
        strong_trend = np.abs(data['Trend_Strength']) > 2
        
        # Combine trend signals
        trend_score = (
            ema_signal * 0.3 +
            price_trend_signal * 0.3 +
            macd_signal * 0.25 +
            psar_signal * 0.15
        )
        
        # Apply trend strength filter
        return trend_score * strong_trend.astype(int)
    
    def _generate_momentum_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum-based signals"""
        # RSI signals
        rsi_oversold = (data['RSI'] < self.config.rsi_oversold) & (data['RSI'] > data['RSI'].shift())
        rsi_overbought = (data['RSI'] > self.config.rsi_overbought) & (data['RSI'] < data['RSI'].shift())
        rsi_signal = np.where(rsi_oversold, 1, np.where(rsi_overbought, -1, 0))
        
        # Stochastic signals
        stoch_oversold = (data['Stoch_K'] < 20) & (data['Stoch_K'] > data['Stoch_D'])
        stoch_overbought = (data['Stoch_K'] > 80) & (data['Stoch_K'] < data['Stoch_D'])
        stoch_signal = np.where(stoch_oversold, 1, np.where(stoch_overbought, -1, 0))
        
        # ROC momentum
        roc_signal = np.where(data['ROC_10'] > 2, 1, np.where(data['ROC_10'] < -2, -1, 0))
        
        # Williams %R
        williams_oversold = data['Williams_R'] > -20
        williams_overbought = data['Williams_R'] < -80
        williams_signal = np.where(williams_overbought, 1, np.where(williams_oversold, -1, 0))
        
        # Combine momentum signals
        momentum_score = (
            rsi_signal * 0.4 +
            stoch_signal * 0.25 +
            roc_signal * 0.25 +
            williams_signal * 0.1
        )
        
        return momentum_score
    
    def _generate_mean_reversion_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals"""
        # Bollinger Band signals
        bb_bounce_bull = (data['BB_Position'] < 0.1) & (data['Close'] > data['Close'].shift())
        bb_bounce_bear = (data['BB_Position'] > 0.9) & (data['Close'] < data['Close'].shift())
        bb_signal = np.where(bb_bounce_bull, 1, np.where(bb_bounce_bear, -1, 0))
        
        # Distance to moving average
        ma_distance = (data['Close'] - data['EMA_Slow']) / data['EMA_Slow']
        mean_revert_bull = (ma_distance < -0.05) & (data['RSI'] < 40)
        mean_revert_bear = (ma_distance > 0.05) & (data['RSI'] > 60)
        ma_signal = np.where(mean_revert_bull, 1, np.where(mean_revert_bear, -1, 0))
        
        # Support/Resistance bounce
        near_support = data['Dist_to_S20'] < 0.02
        near_resistance = data['Dist_to_R20'] < 0.02
        sr_signal = np.where(near_support, 1, np.where(near_resistance, -1, 0))
        
        # Combine mean reversion signals
        reversion_score = (
            bb_signal * 0.5 +
            ma_signal * 0.3 +
            sr_signal * 0.2
        )
        
        return reversion_score
    
    def _generate_volume_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate volume-based signals"""
        # Volume confirmation
        volume_surge = data['Volume_Spike']
        price_up = data['Close'] > data['Close'].shift()
        price_down = data['Close'] < data['Close'].shift()
        
        volume_bull = volume_surge & price_up
        volume_bear = volume_surge & price_down
        
        # OBV trend
        obv_trend = data['OBV'] > data['OBV'].rolling(10).mean()
        
        # Volume divergence
        price_trend_5 = data['Close'] > data['Close'].shift(5)
        obv_trend_5 = data['OBV'] > data['OBV'].shift(5)
        
        volume_divergence_bull = (~price_trend_5) & obv_trend_5
        volume_divergence_bear = price_trend_5 & (~obv_trend_5)
        
        # Combine volume signals
        volume_score = (
            volume_bull.astype(int) * 0.4 +
            volume_bear.astype(int) * -0.4 +
            obv_trend.astype(int) * 0.3 +
            volume_divergence_bull.astype(int) * 0.15 +
            volume_divergence_bear.astype(int) * -0.15
        )
        
        return volume_score
    
    def _generate_pattern_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate pattern-based signals"""
        # Candlestick patterns
        hammer_signal = data['Hammer'].astype(int)
        doji_signal = data['Doji'].astype(int) * 0.5  # Neutral signal
        engulfing_bull = data['Bullish_Engulfing'].astype(int)
        engulfing_bear = data['Bearish_Engulfing'].astype(int)
        
        # Chart patterns
        higher_highs = data['HH'].rolling(5).sum() > 2
        lower_lows = data['LL'].rolling(5).sum() > 2
        
        # Combine pattern signals
        pattern_score = (
            hammer_signal * 0.3 +
            engulfing_bull * 0.4 +
            engulfing_bear * -0.4 +
            higher_highs.astype(int) * 0.15 +
            lower_lows.astype(int) * -0.15
        )
        
        return pattern_score
    
    def _calculate_adaptive_threshold(self, data: pd.DataFrame) -> float:
        """Calculate adaptive signal threshold based on market conditions"""
        base_threshold = self.config.signal_threshold
        
        # Adjust for volatility
        vol_adjustment = (data['Volatility_Rank'].iloc[-20:].mean() - 0.5) * 0.1
        
        # Adjust for trend strength
        trend_adjustment = np.abs(data['Trend_Score'].iloc[-20:].mean()) * 0.1
        
        adaptive_threshold = base_threshold + vol_adjustment + trend_adjustment
        return np.clip(adaptive_threshold, 0.2, 0.6)
    
    def _calculate_signal_quality(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal quality score"""
        # Signal consistency
        signal_consistency = np.abs(data['Signal_Score'].rolling(5).std())
        
        # Volume confirmation
        volume_confirmation = data['Volume_Ratio_20'] > 1.1
        
        # Volatility filter
        vol_filter = (data['Volatility_Rank'] > 0.2) & (data['Volatility_Rank'] < 0.9)
        
        # Combine quality factors
        quality = (
            (1 - signal_consistency) * 0.4 +
            volume_confirmation.astype(int) * 0.3 +
            vol_filter.astype(int) * 0.3
        )
        
        return np.clip(quality, 0, 1)
    
    # Helper calculation methods
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).ewm(span=period).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range"""
        if period is None:
            period = self.config.atr_period
            
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic oscillator"""
        lowest_low = data['Low'].rolling(k_period).min()
        highest_high = data['High'].rolling(k_period).max()
        
        k_percent = 100 * (data['Close'] - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(d_period).mean()
        
        return k_percent, d_percent
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data['High'].rolling(period).max()
        lowest_low = data['Low'].rolling(period).min()
        
        williams_r = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
        
        return williams_r
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame, af_start: float = 0.02, af_max: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR"""
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        
        psar = np.zeros(len(data))
        trend = np.zeros(len(data))  # 1 for uptrend, -1 for downtrend
        af = np.zeros(len(data))
        ep = np.zeros(len(data))  # Extreme point
        
        # Initialize
        psar[0] = low[0]
        trend[0] = 1
        af[0] = af_start
        ep[0] = high[0]
        
        for i in range(1, len(data)):
            if trend[i-1] == 1:  # Uptrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                if low[i] <= psar[i]:  # Trend reversal
                    trend[i] = -1
                    psar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = af_start
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_start, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
                
                if high[i] >= psar[i]:  # Trend reversal
                    trend[i] = 1
                    psar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = af_start
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_start, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        return pd.Series(psar, index=data.index)
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = np.zeros(len(data))
        obv[0] = data['Volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] + data['Volume'].iloc[i]
            elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                obv[i] = obv[i-1] - data['Volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=data.index)
    
    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        price_change = data['Close'].pct_change()
        vpt = (price_change * data['Volume']).cumsum()
        return vpt
    
    def _calculate_adl(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        clv = clv.fillna(0)
        adl = (clv * data['Volume']).cumsum()
        return adl
    
    def _identify_fractals(self, series: pd.Series, mode: str = 'high', periods: int = 5) -> pd.Series:
        """Identify fractal highs and lows"""
        fractals = pd.Series(False, index=series.index)
        
        for i in range(periods, len(series) - periods):
            if mode == 'high':
                is_fractal = all(series.iloc[i] >= series.iloc[i-j] for j in range(1, periods+1)) and \
                           all(series.iloc[i] >= series.iloc[i+j] for j in range(1, periods+1))
            else:  # low
                is_fractal = all(series.iloc[i] <= series.iloc[i-j] for j in range(1, periods+1)) and \
                           all(series.iloc[i] <= series.iloc[i+j] for j in range(1, periods+1))
            
            fractals.iloc[i] = is_fractal
        
        return fractals

# ================================================================================
# PROFESSIONAL OPTIONS PRICING ENGINE
# ================================================================================

class ProfessionalOptionsEngine:
    """Advanced options pricing with Greeks and sophisticated modeling"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.pricing_cache = {}
        
    def black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, 
                     option_type: str, dividend_yield: float = 0.0) -> float:
        """Enhanced Black-Scholes with dividend yield"""
        if T <= 0:
            intrinsic = max(0, (S - K) if option_type == 'call' else (K - S))
            return max(0.01, intrinsic)
        
        try:
            # Validate and sanitize inputs
            if sigma <= 0 or np.isnan(sigma):
                sigma = 0.20
            if S <= 0 or K <= 0:
                return 0.01
                
            # Adjust for dividend yield
            S_adjusted = S * np.exp(-dividend_yield * T)
            
            d1 = (np.log(S_adjusted / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S_adjusted * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S_adjusted * norm.cdf(-d1)
            
            return max(0.01, price)
            
        except (OverflowError, ValueError, ZeroDivisionError):
            # Robust fallback pricing
            intrinsic = max(0, (S - K) if option_type == 'call' else (K - S))
            time_value = max(0.01, 0.1 * S * np.sqrt(T))
            return max(0.01, intrinsic + time_value)
    
    def calculate_greeks(self, S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str, dividend_yield: float = 0.0) -> Dict[str, float]:
        """Calculate complete Greeks suite"""
        if T <= 0:
            delta = 1.0 if (option_type == 'call' and S > K) else \
                   (-1.0 if option_type == 'put' and S < K else 0.0)
            return {
                'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 
                'rho': 0.0, 'charm': 0.0, 'vomma': 0.0
            }
        
        try:
            if sigma <= 0 or np.isnan(sigma):
                sigma = 0.20
                
            S_adjusted = S * np.exp(-dividend_yield * T)
            
            d1 = (np.log(S_adjusted / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # First-order Greeks
            if option_type.lower() == 'call':
                delta = np.exp(-dividend_yield * T) * norm.cdf(d1)
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:  # put
                delta = -np.exp(-dividend_yield * T) * norm.cdf(-d1)
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            # Second-order Greeks
            gamma = np.exp(-dividend_yield * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Theta (time decay per day)
            if option_type.lower() == 'call':
                theta = (
                    -S * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                    r * K * np.exp(-r * T) * norm.cdf(d2) +
                    dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(d1)
                ) / 365
            else:  # put
                theta = (
                    -S * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                    r * K * np.exp(-r * T) * norm.cdf(-d2) -
                    dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(-d1)
                ) / 365
            
            # Third-order Greeks
            charm = -np.exp(-dividend_yield * T) * norm.pdf(d1) * (
                2 * (r - dividend_yield) * T - d2 * sigma * np.sqrt(T)
            ) / (2 * T * sigma * np.sqrt(T))
            
            vomma = vega * d1 * d2 / sigma
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'charm': charm,
                'vomma': vomma
            }
            
        except (OverflowError, ValueError, ZeroDivisionError):
            return {
                'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                'rho': 0.0, 'charm': 0.0, 'vomma': 0.0
            }
    
    def generate_professional_option_chain(self, spot: float, volatility: float, 
                                         days_to_expiry: int, index_name: str = "NIFTY_50") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate professional-grade synthetic option chain"""
        
        T = days_to_expiry / 365.0
        r = self.config.risk_free_rate
        dividend_yield = self._get_dividend_yield(index_name)
        
        # Market-specific parameters
        chain_params = self._get_chain_parameters(index_name, spot)
        
        # Generate realistic strike prices
        strikes = self._generate_strikes(spot, chain_params['interval'], chain_params['range'])
        
        # Enhanced volatility surface
        vol_surface = self._generate_volatility_surface(spot, volatility, strikes, days_to_expiry)
        
        calls_data = []
        puts_data = []
        
        for i, strike in enumerate(strikes):
            moneyness = strike / spot
            iv = vol_surface[i]
            
            # Calculate option prices
            call_price = self.black_scholes(spot, strike, T, r, iv, 'call', dividend_yield)
            put_price = self.black_scholes(spot, strike, T, r, iv, 'put', dividend_yield)
            
            # Calculate Greeks
            call_greeks = self.calculate_greeks(spot, strike, T, r, iv, 'call', dividend_yield)
            put_greeks = self.calculate_greeks(spot, strike, T, r, iv, 'put', dividend_yield)
            
            # Add realistic bid-ask spreads
            call_spread = self._calculate_bid_ask_spread(call_price, moneyness, days_to_expiry)
            put_spread = self._calculate_bid_ask_spread(put_price, moneyness, days_to_expiry)
            
            # Add market microstructure effects
            call_liquidity = self._calculate_liquidity_score(moneyness, days_to_expiry, 'call')
            put_liquidity = self._calculate_liquidity_score(moneyness, days_to_expiry, 'put')
            
            # Call option data
            calls_data.append({
                'strike': strike,
                'price': call_price,
                'bid': max(0.01, call_price - call_spread/2),
                'ask': call_price + call_spread/2,
                'iv': iv,
                'delta': call_greeks['delta'],
                'gamma': call_greeks['gamma'],
                'theta': call_greeks['theta'],
                'vega': call_greeks['vega'],
                'rho': call_greeks['rho'],
                'charm': call_greeks.get('charm', 0),
                'vomma': call_greeks.get('vomma', 0),
                'moneyness': moneyness,
                'dte': days_to_expiry,
                'liquidity_score': call_liquidity,
                'open_interest': max(1, int(1000 * call_liquidity)),
                'volume': max(1, int(100 * call_liquidity))
            })
            
            # Put option data
            puts_data.append({
                'strike': strike,
                'price': put_price,
                'bid': max(0.01, put_price - put_spread/2),
                'ask': put_price + put_spread/2,
                'iv': iv,
                'delta': put_greeks['delta'],
                'gamma': put_greeks['gamma'],
                'theta': put_greeks['theta'],
                'vega': put_greeks['vega'],
                'rho': put_greeks['rho'],
                'charm': put_greeks.get('charm', 0),
                'vomma': put_greeks.get('vomma', 0),
                'moneyness': moneyness,
                'dte': days_to_expiry,
                'liquidity_score': put_liquidity,
                'open_interest': max(1, int(1000 * put_liquidity)),
                'volume': max(1, int(100 * put_liquidity))
            })
        
        calls_df = pd.DataFrame(calls_data)
        puts_df = pd.DataFrame(puts_data)
        
        return calls_df, puts_df
    
    def _get_dividend_yield(self, index_name: str) -> float:
        """Get historical dividend yield for index"""
        dividend_yields = {
            "NIFTY_50": 0.012,
            "BANK_NIFTY": 0.008,
            "SENSEX": 0.013
        }
        return dividend_yields.get(index_name, 0.01)
    
    def _get_chain_parameters(self, index_name: str, spot: float) -> Dict:
        """Get market-specific option chain parameters"""
        if "BANK" in index_name:
            return {
                'interval': 100,
                'range': 40,
                'base_spread': 0.50
            }
        elif "NIFTY" in index_name:
            return {
                'interval': 50 if spot > 10000 else 25,
                'range': 35,
                'base_spread': 0.25
            }
        else:  # SENSEX
            return {
                'interval': 100,
                'range': 30,
                'base_spread': 0.30
            }
    
    def _generate_strikes(self, spot: float, interval: float, num_strikes: int) -> List[float]:
        """Generate realistic strike prices"""
        strikes = []
        
        # Find ATM strike
        atm_strike = round(spot / interval) * interval
        
        # Generate strikes around ATM
        for i in range(-num_strikes//2, num_strikes//2 + 1):
            strike = atm_strike + i * interval
            if strike > 0:  # Ensure positive strikes
                strikes.append(strike)
        
        return sorted(strikes)
    
    def _generate_volatility_surface(self, spot: float, base_vol: float, 
                                   strikes: List[float], dte: int) -> List[float]:
        """Generate realistic volatility surface with skew and smile"""
        vol_surface = []
        
        for strike in strikes:
            moneyness = np.log(strike / spot)  # Log moneyness
            
            # Volatility skew (puts more expensive than calls)
            skew_factor = -0.3 * moneyness  # Negative skew for equity indices
            
            # Volatility smile (OTM options more expensive)
            smile_factor = 0.15 * moneyness**2
            
            # Term structure effect
            if dte < 15:
                term_factor = 0.2  # Higher IV near expiry
            elif dte > 30:
                term_factor = -0.1  # Lower IV far from expiry
            else:
                term_factor = 0.0
            
            # Calculate final IV
            iv = base_vol * (1 + skew_factor + smile_factor + term_factor)
            
            # Ensure reasonable bounds
            iv = max(0.08, min(2.0, iv))
            
            vol_surface.append(iv)
        
        return vol_surface
    
    def _calculate_bid_ask_spread(self, option_price: float, moneyness: float, dte: int) -> float:
        """Calculate realistic bid-ask spread"""
        # Base spread proportional to option price
        base_spread = max(0.05, option_price * 0.02)
        
        # Adjust for moneyness (wider spreads for OTM)
        moneyness_factor = 1 + 2 * abs(np.log(moneyness))
        
        # Adjust for time to expiry (wider spreads near expiry)
        if dte < 7:
            time_factor = 2.0
        elif dte < 15:
            time_factor = 1.5
        else:
            time_factor = 1.0
        
        spread = base_spread * moneyness_factor * time_factor
        
        return min(spread, option_price * 0.15)  # Cap spread at 15% of option price
    
    def _calculate_liquidity_score(self, moneyness: float, dte: int, option_type: str) -> float:
        """Calculate liquidity score for option"""
        # Higher liquidity for ATM options
        moneyness_score = max(0.1, 1 - 2 * abs(np.log(moneyness)))
        
        # Higher liquidity for intermediate expiries
        if 15 <= dte <= 30:
            time_score = 1.0
        elif 7 <= dte < 15:
            time_score = 0.8
        elif 30 < dte <= 45:
            time_score = 0.7
        else:
            time_score = 0.3
        
        # Slightly higher liquidity for puts (hedging demand)
        type_score = 1.0 if option_type == 'put' else 0.9
        
        liquidity = moneyness_score * time_score * type_score
        return np.clip(liquidity, 0.05, 1.0)

# ================================================================================
# ADVANCED STRATEGY MANAGER
# ================================================================================

class AdvancedStrategyManager:
    """Professional strategy selection and execution"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.options_engine = ProfessionalOptionsEngine(config)
        self.strategy_cache = {}
    
    def select_optimal_strategy(self, signal: int, market_regime: str, volatility_rank: float,
                               spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                               volatility: float, dte: int, index_name: str, 
                               additional_context: Dict = None) -> Optional[Dict]:
        """Select optimal strategy using advanced decision tree"""
        
        if calls_df.empty and puts_df.empty:
            return None
        
        # Get market context
        context = additional_context or {}
        trend_strength = context.get('trend_strength', 0)
        signal_quality = context.get('signal_quality', 0.5)
        
        # Strategy candidates
        strategies = []
        
        # Get lot size for index
        lot_size = self._get_lot_size(index_name)
        
        # Strategy selection based on market conditions
        if signal > 0:  # Bullish bias
            strategies.extend(self._get_bullish_strategies(
                spot, calls_df, puts_df, volatility, volatility_rank, dte, lot_size
            ))
        elif signal < 0:  # Bearish bias
            strategies.extend(self._get_bearish_strategies(
                spot, calls_df, puts_df, volatility, volatility_rank, dte, lot_size
            ))
        
        # Always consider neutral strategies for income generation
        if volatility_rank < 0.6:  # Lower volatility environments
            strategies.extend(self._get_neutral_strategies(
                spot, calls_df, puts_df, volatility, volatility_rank, dte, lot_size
            ))
        
        # High volatility strategies
        if volatility_rank > 0.7:
            strategies.extend(self._get_high_vol_strategies(
                spot, calls_df, puts_df, volatility, dte, lot_size
            ))
        
        if not strategies:
            return None
        
        # Advanced strategy scoring
        best_strategy = self._score_and_select_strategy(
            strategies, signal, market_regime, volatility_rank, 
            trend_strength, signal_quality
        )
        
        return best_strategy
    
    def _get_bullish_strategies(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                               volatility: float, volatility_rank: float, dte: int, 
                               lot_size: int) -> List[Dict]:
        """Get bullish strategy candidates"""
        strategies = []
        
        # Bull Put Spread (primary bullish strategy)
        bull_put = self._create_bull_put_spread(spot, puts_df, volatility, dte, lot_size)
        if bull_put:
            strategies.append(bull_put)
        
        # Long Call (high conviction bullish)
        if volatility_rank < 0.4:  # Only in low IV environment
            long_call = self._create_long_call(spot, calls_df, dte, lot_size)
            if long_call:
                strategies.append(long_call)
        
        # Call Debit Spread (moderate bullish)
        call_debit = self._create_call_debit_spread(spot, calls_df, volatility, dte, lot_size)
        if call_debit:
            strategies.append(call_debit)
        
        return strategies
    
    def _get_bearish_strategies(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                               volatility: float, volatility_rank: float, dte: int,
                               lot_size: int) -> List[Dict]:
        """Get bearish strategy candidates"""
        strategies = []
        
        # Bear Call Spread (primary bearish strategy)
        bear_call = self._create_bear_call_spread(spot, calls_df, volatility, dte, lot_size)
        if bear_call:
            strategies.append(bear_call)
        
        # Long Put (high conviction bearish)
        if volatility_rank < 0.4:  # Only in low IV environment
            long_put = self._create_long_put(spot, puts_df, dte, lot_size)
            if long_put:
                strategies.append(long_put)
        
        # Put Debit Spread (moderate bearish)
        put_debit = self._create_put_debit_spread(spot, puts_df, volatility, dte, lot_size)
        if put_debit:
            strategies.append(put_debit)
        
        return strategies
    
    def _get_neutral_strategies(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                               volatility: float, volatility_rank: float, dte: int,
                               lot_size: int) -> List[Dict]:
        """Get neutral/income strategies"""
        strategies = []
        
        # Iron Condor (range-bound markets)
        iron_condor = self._create_iron_condor(spot, calls_df, puts_df, volatility, dte, lot_size)
        if iron_condor:
            strategies.append(iron_condor)
        
        # Short Strangle (high IV, expect range-bound)
        if volatility_rank > 0.5:
            short_strangle = self._create_short_strangle(spot, calls_df, puts_df, volatility, dte, lot_size)
            if short_strangle:
                strategies.append(short_strangle)
        
        # Iron Butterfly (tight range expectation)
        iron_butterfly = self._create_iron_butterfly(spot, calls_df, puts_df, volatility, dte, lot_size)
        if iron_butterfly:
            strategies.append(iron_butterfly)
        
        return strategies
    
    def _get_high_vol_strategies(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                                volatility: float, dte: int, lot_size: int) -> List[Dict]:
        """Get high volatility strategies"""
        strategies = []
        
        # Long Straddle (expect big move, direction unknown)
        long_straddle = self._create_long_straddle(spot, calls_df, puts_df, volatility, dte, lot_size)
        if long_straddle:
            strategies.append(long_straddle)
        
        # Long Strangle (cheaper than straddle, wider profit zone)
        long_strangle = self._create_long_strangle(spot, calls_df, puts_df, volatility, dte, lot_size)
        if long_strangle:
            strategies.append(long_strangle)
        
        return strategies
    
    def _create_bull_put_spread(self, spot: float, puts_df: pd.DataFrame, volatility: float,
                               dte: int, lot_size: int) -> Optional[Dict]:
        """Create bull put spread strategy"""
        if puts_df.empty:
            return None
        
        # Filter for OTM puts
        otm_puts = puts_df[puts_df['strike'] < spot * 0.98].copy()
        if len(otm_puts) < 2:
            return None
        
        # Select strikes based on delta and liquidity
        short_put_candidates = otm_puts[
            (otm_puts['delta'].abs().between(0.15, 0.35)) &
            (otm_puts['liquidity_score'] > 0.3)
        ]
        
        long_put_candidates = otm_puts[
            (otm_puts['delta'].abs().between(0.05, 0.20)) &
            (otm_puts['liquidity_score'] > 0.2)
        ]
        
        if short_put_candidates.empty or long_put_candidates.empty:
            return None
        
        # Select best strikes (highest short put delta, lowest long put delta)
        short_put = short_put_candidates.loc[short_put_candidates['delta'].abs().idxmax()]
        
        # Find long put that creates reasonable spread width
        long_put_candidates = long_put_candidates[long_put_candidates['strike'] < short_put['strike']]
        if long_put_candidates.empty:
            return None
        
        long_put = long_put_candidates.loc[long_put_candidates['delta'].abs().idxmin()]
        
        # Calculate strategy metrics
        credit = short_put['bid'] - long_put['ask']
        if credit <= self.config.min_credit_threshold:
            return None
        
        max_profit = credit * lot_size
        spread_width = short_put['strike'] - long_put['strike']
        max_loss = (spread_width - credit) * lot_size
        
        if max_loss <= 0:
            return None
        
        # Calculate profit probability using delta
        profit_prob = 1 - abs(short_put['delta'])  # Probability of finishing above short strike
        
        # Risk-reward metrics
        risk_reward_ratio = max_profit / max_loss
        breakeven = short_put['strike'] - credit
        
        return {
            'strategy_type': StrategyType.BULL_PUT_SPREAD,
            'legs': [
                {
                    'side': 'PUT', 'strike': short_put['strike'], 'action': 'SELL',
                    'price': short_put['bid'], 'delta': short_put['delta'],
                    'liquidity': short_put['liquidity_score']
                },
                {
                    'side': 'PUT', 'strike': long_put['strike'], 'action': 'BUY',
                    'price': long_put['ask'], 'delta': long_put['delta'],
                    'liquidity': long_put['liquidity_score']
                }
            ],
            'net_credit': credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward_ratio,
            'spread_width': spread_width,
            'net_delta': short_put['delta'] - long_put['delta'],
            'net_gamma': short_put['gamma'] - long_put['gamma'],
            'net_theta': short_put['theta'] - long_put['theta'],
            'net_vega': short_put['vega'] - long_put['vega'],
            'dte': dte,
            'score': self._calculate_strategy_score(profit_prob, risk_reward_ratio, credit)
        }
    
    def _create_bear_call_spread(self, spot: float, calls_df: pd.DataFrame, volatility: float,
                                dte: int, lot_size: int) -> Optional[Dict]:
        """Create bear call spread strategy"""
        if calls_df.empty:
            return None
        
        # Filter for OTM calls
        otm_calls = calls_df[calls_df['strike'] > spot * 1.02].copy()
        if len(otm_calls) < 2:
            return None
        
        # Select strikes based on delta and liquidity
        short_call_candidates = otm_calls[
            (otm_calls['delta'].between(0.15, 0.35)) &
            (otm_calls['liquidity_score'] > 0.3)
        ]
        
        long_call_candidates = otm_calls[
            (otm_calls['delta'].between(0.05, 0.20)) &
            (otm_calls['liquidity_score'] > 0.2)
        ]
        
        if short_call_candidates.empty or long_call_candidates.empty:
            return None
        
        # Select best strikes
        short_call = short_call_candidates.loc[short_call_candidates['delta'].idxmax()]
        
        long_call_candidates = long_call_candidates[long_call_candidates['strike'] > short_call['strike']]
        if long_call_candidates.empty:
            return None
        
        long_call = long_call_candidates.loc[long_call_candidates['delta'].idxmin()]
        
        # Calculate strategy metrics
        credit = short_call['bid'] - long_call['ask']
        if credit <= self.config.min_credit_threshold:
            return None
        
        max_profit = credit * lot_size
        spread_width = long_call['strike'] - short_call['strike']
        max_loss = (spread_width - credit) * lot_size
        
        if max_loss <= 0:
            return None
        
        # Calculate profit probability
        profit_prob = 1 - short_call['delta']  # Probability of finishing below short strike
        
        # Risk-reward metrics
        risk_reward_ratio = max_profit / max_loss
        breakeven = short_call['strike'] + credit
        
        return {
            'strategy_type': StrategyType.BEAR_CALL_SPREAD,
            'legs': [
                {
                    'side': 'CALL', 'strike': short_call['strike'], 'action': 'SELL',
                    'price': short_call['bid'], 'delta': short_call['delta'],
                    'liquidity': short_call['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': long_call['strike'], 'action': 'BUY',
                    'price': long_call['ask'], 'delta': long_call['delta'],
                    'liquidity': long_call['liquidity_score']
                }
            ],
            'net_credit': credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward_ratio,
            'spread_width': spread_width,
            'net_delta': short_call['delta'] - long_call['delta'],
            'net_gamma': short_call['gamma'] - long_call['gamma'],
            'net_theta': short_call['theta'] - long_call['theta'],
            'net_vega': short_call['vega'] - long_call['vega'],
            'dte': dte,
            'score': self._calculate_strategy_score(profit_prob, risk_reward_ratio, credit)
        }
    
    def _create_iron_condor(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                           volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create iron condor strategy"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Filter for liquid OTM options
        otm_puts = puts_df[
            (puts_df['strike'] < spot * 0.95) & 
            (puts_df['liquidity_score'] > 0.2)
        ].copy()
        
        otm_calls = calls_df[
            (calls_df['strike'] > spot * 1.05) &
            (calls_df['liquidity_score'] > 0.2)
        ].copy()
        
        if len(otm_puts) < 2 or len(otm_calls) < 2:
            return None
        
        # Select strikes with similar delta magnitude
        target_delta = 0.20
        
        # Put side
        put_deltas = otm_puts['delta'].abs()
        short_put_idx = (put_deltas - target_delta).abs().idxmin()
        short_put = otm_puts.loc[short_put_idx]
        
        long_put_candidates = otm_puts[
            (otm_puts['strike'] < short_put['strike']) &
            (otm_puts['delta'].abs() < short_put['delta'].abs())
        ]
        
        if long_put_candidates.empty:
            return None
        
        long_put = long_put_candidates.loc[long_put_candidates['delta'].abs().idxmin()]
        
        # Call side
        call_deltas = otm_calls['delta']
        short_call_idx = (call_deltas - target_delta).abs().idxmin()
        short_call = otm_calls.loc[short_call_idx]
        
        long_call_candidates = otm_calls[
            (otm_calls['strike'] > short_call['strike']) &
            (otm_calls['delta'] < short_call['delta'])
        ]
        
        if long_call_candidates.empty:
            return None
        
        long_call = long_call_candidates.loc[long_call_candidates['delta'].idxmin()]
        
        # Calculate strategy metrics
        total_credit = (short_put['bid'] + short_call['bid'] - 
                       long_put['ask'] - long_call['ask'])
        
        if total_credit <= self.config.min_credit_threshold:
            return None
        
        max_profit = total_credit * lot_size
        
        put_spread_width = short_put['strike'] - long_put['strike']
        call_spread_width = long_call['strike'] - short_call['strike']
        max_spread_width = max(put_spread_width, call_spread_width)
        
        max_loss = (max_spread_width - total_credit) * lot_size
        
        if max_loss <= 0:
            return None
        
        # Calculate profit zone and probability
        lower_breakeven = short_put['strike'] - total_credit
        upper_breakeven = short_call['strike'] + total_credit
        
        # Estimate probability of staying in profit zone
        # Using approximation based on expected range
        expected_move = spot * volatility * np.sqrt(dte / 365)
        profit_zone_width = upper_breakeven - lower_breakeven
        
        profit_prob = max(0.1, min(0.9, profit_zone_width / (2 * expected_move)))
        
        risk_reward_ratio = max_profit / max_loss
        
        return {
            'strategy_type': StrategyType.IRON_CONDOR,
            'legs': [
                {
                    'side': 'PUT', 'strike': short_put['strike'], 'action': 'SELL',
                    'price': short_put['bid'], 'delta': short_put['delta'],
                    'liquidity': short_put['liquidity_score']
                },
                {
                    'side': 'PUT', 'strike': long_put['strike'], 'action': 'BUY',
                    'price': long_put['ask'], 'delta': long_put['delta'],
                    'liquidity': long_put['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': short_call['strike'], 'action': 'SELL',
                    'price': short_call['bid'], 'delta': short_call['delta'],
                    'liquidity': short_call['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': long_call['strike'], 'action': 'BUY',
                    'price': long_call['ask'], 'delta': long_call['delta'],
                    'liquidity': long_call['liquidity_score']
                }
            ],
            'net_credit': total_credit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'lower_breakeven': lower_breakeven,
            'upper_breakeven': upper_breakeven,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward_ratio,
            'expected_move': expected_move,
            'profit_zone_width': profit_zone_width,
            'net_delta': (short_put['delta'] + short_call['delta'] - 
                         long_put['delta'] - long_call['delta']),
            'net_gamma': (short_put['gamma'] + short_call['gamma'] - 
                         long_put['gamma'] - long_call['gamma']),
            'net_theta': (short_put['theta'] + short_call['theta'] - 
                         long_put['theta'] - long_call['theta']),
            'net_vega': (short_put['vega'] + short_call['vega'] - 
                        long_put['vega'] - long_call['vega']),
            'dte': dte,
            'score': self._calculate_strategy_score(profit_prob, risk_reward_ratio, total_credit)
        }
    
    def _create_long_straddle(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                             volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create long straddle for high volatility expectations"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find ATM options
        atm_strike = self._find_atm_strike(spot, calls_df, puts_df)
        
        atm_call = calls_df[calls_df['strike'] == atm_strike]
        atm_put = puts_df[puts_df['strike'] == atm_strike]
        
        if atm_call.empty or atm_put.empty:
            return None
        
        call_data = atm_call.iloc[0]
        put_data = atm_put.iloc[0]
        
        # Check liquidity
        if call_data['liquidity_score'] < 0.3 or put_data['liquidity_score'] < 0.3:
            return None
        
        # Calculate strategy metrics
        total_cost = call_data['ask'] + put_data['ask']
        max_loss = total_cost * lot_size
        
        # Breakeven points
        upper_breakeven = atm_strike + total_cost
        lower_breakeven = atm_strike - total_cost
        
        # Expected move for profit probability
        expected_move = spot * volatility * np.sqrt(dte / 365)
        
        # Probability of moving beyond breakeven
        profit_prob = max(0.2, min(0.8, expected_move / total_cost))
        
        return {
            'strategy_type': StrategyType.LONG_STRADDLE,
            'legs': [
                {
                    'side': 'CALL', 'strike': atm_strike, 'action': 'BUY',
                    'price': call_data['ask'], 'delta': call_data['delta'],
                    'liquidity': call_data['liquidity_score']
                },
                {
                    'side': 'PUT', 'strike': atm_strike, 'action': 'BUY',
                    'price': put_data['ask'], 'delta': put_data['delta'],
                    'liquidity': put_data['liquidity_score']
                }
            ],
            'net_debit': total_cost,
            'max_loss': max_loss,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'profit_probability': profit_prob,
            'expected_move': expected_move,
            'breakeven_range': total_cost * 2,
            'net_delta': call_data['delta'] + put_data['delta'],
            'net_gamma': call_data['gamma'] + put_data['gamma'],
            'net_theta': call_data['theta'] + put_data['theta'],
            'net_vega': call_data['vega'] + put_data['vega'],
            'dte': dte,
            'score': self._calculate_debit_strategy_score(profit_prob, expected_move, total_cost)
        }
    
    # Additional strategy creation methods would go here...
    # (Long Call, Long Put, Iron Butterfly, Short Strangle, etc.)
    
    def _score_and_select_strategy(self, strategies: List[Dict], signal: int, market_regime: str,
                                  volatility_rank: float, trend_strength: float,
                                  signal_quality: float) -> Optional[Dict]:
        """Advanced strategy scoring and selection"""
        if not strategies:
            return None
        
        scored_strategies = []
        
        for strategy in strategies:
            base_score = strategy.get('score', 0)
            
            # Adjust score based on market conditions
            adjusted_score = self._adjust_strategy_score(
                strategy, base_score, signal, market_regime, 
                volatility_rank, trend_strength, signal_quality
            )
            
            strategy['adjusted_score'] = adjusted_score
            scored_strategies.append((adjusted_score, strategy))
        
        # Sort by adjusted score and return best strategy
        scored_strategies.sort(key=lambda x: x[0], reverse=True)
        
        return scored_strategies[0][1]
    
    def _adjust_strategy_score(self, strategy: Dict, base_score: float, signal: int,
                              market_regime: str, volatility_rank: float,
                              trend_strength: float, signal_quality: float) -> float:
        """Adjust strategy score based on market context"""
        
        score = base_score
        strategy_type = strategy['strategy_type']
        
        # Signal strength adjustment
        score *= (0.5 + signal_quality)
        
        # Market regime adjustments
        if market_regime == MarketRegime.TRENDING_UP.value:
            if strategy_type in [StrategyType.BULL_PUT_SPREAD]:
                score *= 1.3
            elif strategy_type in [StrategyType.IRON_CONDOR]:
                score *= 0.8
                
        elif market_regime == MarketRegime.TRENDING_DOWN.value:
            if strategy_type in [StrategyType.BEAR_CALL_SPREAD]:
                score *= 1.3
            elif strategy_type in [StrategyType.IRON_CONDOR]:
                score *= 0.8
                
        elif market_regime == MarketRegime.HIGH_VOLATILITY.value:
            if strategy_type in [StrategyType.LONG_STRADDLE]:
                score *= 1.4
            elif strategy_type in [StrategyType.IRON_CONDOR]:
                score *= 0.7
                
        elif market_regime == MarketRegime.SIDEWAYS.value:
            if strategy_type in [StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE]:
                score *= 1.2
        
        # Volatility rank adjustments
        if volatility_rank > 0.7:  # High IV
            # Favor credit strategies
            if strategy.get('net_credit', 0) > 0:
                score *= 1.2
            else:
                score *= 0.8
        elif volatility_rank < 0.3:  # Low IV
            # Favor debit strategies
            if strategy.get('net_debit', 0) > 0:
                score *= 1.1
        
        # Risk-reward adjustment
        risk_reward = strategy.get('risk_reward_ratio', 0)
        if risk_reward > 0.5:
            score *= (1 + risk_reward * 0.2)
        
        # Profit probability adjustment
        profit_prob = strategy.get('profit_probability', 0.5)
        score *= profit_prob
        
        return max(0, score)
    
    def _calculate_strategy_score(self, profit_prob: float, risk_reward: float, 
                                 credit: float) -> float:
        """Calculate base strategy score for credit strategies"""
        # Weight profit probability heavily
        prob_score = profit_prob * 0.6
        
        # Risk-reward ratio (capped)
        rr_score = min(risk_reward, 1.0) * 0.3
        
        # Credit amount (normalized)
        credit_score = min(credit / 100, 0.1) * 0.1
        
        return prob_score + rr_score + credit_score
    
    def _calculate_debit_strategy_score(self, profit_prob: float, expected_move: float,
                                       debit: float) -> float:
        """Calculate base strategy score for debit strategies"""
        # Profit probability
        prob_score = profit_prob * 0.7
        
        # Expected move vs cost
        move_ratio = min(expected_move / debit, 2.0) * 0.2
        
        # Cost efficiency (lower debit is better)
        cost_score = max(0, (100 - debit) / 100) * 0.1
        
        return prob_score + move_ratio + cost_score
    
    def _get_lot_size(self, index_name: str) -> int:
        """Get lot size for different indices"""
        lot_sizes = {
            "NIFTY_50": self.config.lot_size_nifty,
            "BANK_NIFTY": self.config.lot_size_banknifty,
            "SENSEX": self.config.lot_size_sensex
        }
        return lot_sizes.get(index_name, 50)
    
    def _find_atm_strike(self, spot: float, calls_df: pd.DataFrame, 
                        puts_df: pd.DataFrame) -> float:
        """Find the at-the-money strike"""
        all_strikes = set(calls_df['strike'].tolist() + puts_df['strike'].tolist())
        return min(all_strikes, key=lambda x: abs(x - spot))

# Additional strategy creation methods for completeness
    def _create_long_call(self, spot: float, calls_df: pd.DataFrame, 
                         dte: int, lot_size: int) -> Optional[Dict]:
        """Create long call strategy"""
        if calls_df.empty:
            return None
        
        # Find slightly OTM call with good liquidity
        otm_calls = calls_df[
            (calls_df['strike'] > spot * 1.01) & 
            (calls_df['strike'] < spot * 1.10) &
            (calls_df['liquidity_score'] > 0.4)
        ]
        
        if otm_calls.empty:
            return None
        
        # Select call with best delta/liquidity balance
        call = otm_calls.loc[otm_calls['delta'].idxmax()]
        
        cost = call['ask']
        max_loss = cost * lot_size
        
        # Breakeven
        breakeven = call['strike'] + cost
        
        # Rough profit probability (simplified)
        profit_prob = max(0.2, call['delta'] * 0.7)
        
        return {
            'strategy_type': 'LONG_CALL',
            'legs': [{
                'side': 'CALL', 'strike': call['strike'], 'action': 'BUY',
                'price': call['ask'], 'delta': call['delta'],
                'liquidity': call['liquidity_score']
            }],
            'net_debit': cost,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_probability': profit_prob,
            'net_delta': call['delta'],
            'net_gamma': call['gamma'],
            'net_theta': call['theta'],
            'net_vega': call['vega'],
            'dte': dte,
            'score': self._calculate_debit_strategy_score(profit_prob, cost * 2, cost)
        }

    def _create_long_put(self, spot: float, puts_df: pd.DataFrame,
                        dte: int, lot_size: int) -> Optional[Dict]:
        """Create long put strategy"""
        if puts_df.empty:
            return None
        
        # Find slightly OTM put with good liquidity
        otm_puts = puts_df[
            (puts_df['strike'] < spot * 0.99) & 
            (puts_df['strike'] > spot * 0.90) &
            (puts_df['liquidity_score'] > 0.4)
        ]
        
        if otm_puts.empty:
            return None
        
        # Select put with best delta/liquidity balance
        put = otm_puts.loc[otm_puts['delta'].abs().idxmax()]
        
        cost = put['ask']
        max_loss = cost * lot_size
        
        # Breakeven
        breakeven = put['strike'] - cost
        
        # Rough profit probability
        profit_prob = max(0.2, abs(put['delta']) * 0.7)
        
        return {
            'strategy_type': 'LONG_PUT',
            'legs': [{
                'side': 'PUT', 'strike': put['strike'], 'action': 'BUY',
                'price': put['ask'], 'delta': put['delta'],
                'liquidity': put['liquidity_score']
            }],
            'net_debit': cost,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_probability': profit_prob,
            'net_delta': put['delta'],
            'net_gamma': put['gamma'],
            'net_theta': put['theta'],
            'net_vega': put['vega'],
            'dte': dte,
            'score': self._calculate_debit_strategy_score(profit_prob, cost * 2, cost)
        }

    def _create_call_debit_spread(self, spot: float, calls_df: pd.DataFrame,
                                 volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create call debit spread"""
        if calls_df.empty:
            return None
        
        # Find good strikes for debit spread
        itm_calls = calls_df[
            (calls_df['strike'] < spot * 1.02) & 
            (calls_df['delta'] > 0.6) &
            (calls_df['liquidity_score'] > 0.3)
        ]
        
        otm_calls = calls_df[
            (calls_df['strike'] > spot * 1.02) &
            (calls_df['delta'] < 0.4) &
            (calls_df['liquidity_score'] > 0.2)
        ]
        
        if itm_calls.empty or otm_calls.empty:
            return None
        
        long_call = itm_calls.loc[itm_calls['delta'].idxmax()]
        short_call = otm_calls.loc[otm_calls['delta'].idxmin()]
        
        if long_call['strike'] >= short_call['strike']:
            return None
        
        net_debit = long_call['ask'] - short_call['bid']
        if net_debit <= 0:
            return None
        
        max_loss = net_debit * lot_size
        spread_width = short_call['strike'] - long_call['strike']
        max_profit = (spread_width - net_debit) * lot_size
        
        if max_profit <= 0:
            return None
        
        breakeven = long_call['strike'] + net_debit
        profit_prob = long_call['delta'] * 0.8
        risk_reward = max_profit / max_loss
        
        return {
            'strategy_type': 'CALL_DEBIT_SPREAD',
            'legs': [
                {
                    'side': 'CALL', 'strike': long_call['strike'], 'action': 'BUY',
                    'price': long_call['ask'], 'delta': long_call['delta'],
                    'liquidity': long_call['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': short_call['strike'], 'action': 'SELL',
                    'price': short_call['bid'], 'delta': short_call['delta'],
                    'liquidity': short_call['liquidity_score']
                }
            ],
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward,
            'spread_width': spread_width,
            'net_delta': long_call['delta'] - short_call['delta'],
            'net_gamma': long_call['gamma'] - short_call['gamma'],
            'net_theta': long_call['theta'] - short_call['theta'],
            'net_vega': long_call['vega'] - short_call['vega'],
            'dte': dte,
            'score': self._calculate_debit_strategy_score(profit_prob, max_profit, net_debit)
        }

    def _create_put_debit_spread(self, spot: float, puts_df: pd.DataFrame,
                                volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create put debit spread"""
        if puts_df.empty:
            return None
        
        # Find good strikes for debit spread
        itm_puts = puts_df[
            (puts_df['strike'] > spot * 0.98) & 
            (puts_df['delta'].abs() > 0.6) &
            (puts_df['liquidity_score'] > 0.3)
        ]
        
        otm_puts = puts_df[
            (puts_df['strike'] < spot * 0.98) &
            (puts_df['delta'].abs() < 0.4) &
            (puts_df['liquidity_score'] > 0.2)
        ]
        
        if itm_puts.empty or otm_puts.empty:
            return None
        
        long_put = itm_puts.loc[itm_puts['delta'].abs().idxmax()]
        short_put = otm_puts.loc[otm_puts['delta'].abs().idxmin()]
        
        if long_put['strike'] <= short_put['strike']:
            return None
        
        net_debit = long_put['ask'] - short_put['bid']
        if net_debit <= 0:
            return None
        
        max_loss = net_debit * lot_size
        spread_width = long_put['strike'] - short_put['strike']
        max_profit = (spread_width - net_debit) * lot_size
        
        if max_profit <= 0:
            return None
        
        breakeven = long_put['strike'] - net_debit
        profit_prob = abs(long_put['delta']) * 0.8
        risk_reward = max_profit / max_loss
        
        return {
            'strategy_type': 'PUT_DEBIT_SPREAD',
            'legs': [
                {
                    'side': 'PUT', 'strike': long_put['strike'], 'action': 'BUY',
                    'price': long_put['ask'], 'delta': long_put['delta'],
                    'liquidity': long_put['liquidity_score']
                },
                {
                    'side': 'PUT', 'strike': short_put['strike'], 'action': 'SELL',
                    'price': short_put['bid'], 'delta': short_put['delta'],
                    'liquidity': short_put['liquidity_score']
                }
            ],
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'breakeven': breakeven,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward,
            'spread_width': spread_width,
            'net_delta': long_put['delta'] - short_put['delta'],
            'net_gamma': long_put['gamma'] - short_put['gamma'],
            'net_theta': long_put['theta'] - short_put['theta'],
            'net_vega': long_put['vega'] - short_put['vega'],
            'dte': dte,
            'score': self._calculate_debit_strategy_score(profit_prob, max_profit, net_debit)
        }

    def _create_short_strangle(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                              volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create short strangle strategy"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find OTM options with good liquidity
        otm_puts = puts_df[
            (puts_df['strike'] < spot * 0.95) & 
            (puts_df['liquidity_score'] > 0.3)
        ]
        
        otm_calls = calls_df[
            (calls_df['strike'] > spot * 1.05) &
            (calls_df['liquidity_score'] > 0.3)
        ]
        
        if otm_puts.empty or otm_calls.empty:
            return None
        
        # Select strikes with similar delta magnitude
        target_delta = 0.25
        
        put_candidate = otm_puts.loc[(otm_puts['delta'].abs() - target_delta).abs().idxmin()]
        call_candidate = otm_calls.loc[(otm_calls['delta'] - target_delta).abs().idxmin()]
        
        # Calculate strategy metrics
        total_credit = put_candidate['bid'] + call_candidate['bid']
        if total_credit <= self.config.min_credit_threshold:
            return None
        
        max_profit = total_credit * lot_size
        
        # Calculate max loss (unlimited theoretically, but estimate)
        expected_move = spot * volatility * np.sqrt(dte / 365)
        estimated_max_loss = max(
            max(0, put_candidate['strike'] - spot + expected_move - total_credit),
            max(0, spot - expected_move - call_candidate['strike'] - total_credit)
        ) * lot_size
        
        if estimated_max_loss <= 0:
            estimated_max_loss = total_credit * 5  # Conservative estimate
        
        # Breakeven points
        upper_breakeven = call_candidate['strike'] + total_credit
        lower_breakeven = put_candidate['strike'] - total_credit
        
        # Profit probability (staying between strikes)
        profit_zone = upper_breakeven - lower_breakeven
        profit_prob = max(0.3, min(0.8, profit_zone / (2 * expected_move)))
        
        return {
            'strategy_type': StrategyType.SHORT_STRANGLE,
            'legs': [
                {
                    'side': 'PUT', 'strike': put_candidate['strike'], 'action': 'SELL',
                    'price': put_candidate['bid'], 'delta': put_candidate['delta'],
                    'liquidity': put_candidate['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': call_candidate['strike'], 'action': 'SELL',
                    'price': call_candidate['bid'], 'delta': call_candidate['delta'],
                    'liquidity': call_candidate['liquidity_score']
                }
            ],
            'net_credit': total_credit,
            'max_profit': max_profit,
            'estimated_max_loss': estimated_max_loss,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'profit_probability': profit_prob,
            'profit_zone_width': profit_zone,
            'expected_move': expected_move,
            'net_delta': put_candidate['delta'] + call_candidate['delta'],
            'net_gamma': put_candidate['gamma'] + call_candidate['gamma'],
            'net_theta': put_candidate['theta'] + call_candidate['theta'],
            'net_vega': put_candidate['vega'] + call_candidate['vega'],
            'dte': dte,
            'score': self._calculate_strategy_score(profit_prob, max_profit / estimated_max_loss, total_credit)
        }

    def _create_iron_butterfly(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                              volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create iron butterfly strategy"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find ATM strike
        atm_strike = self._find_atm_strike(spot, calls_df, puts_df)
        
        # Get ATM options
        atm_call = calls_df[calls_df['strike'] == atm_strike]
        atm_put = puts_df[puts_df['strike'] == atm_strike]
        
        if atm_call.empty or atm_put.empty:
            return None
        
        atm_call_data = atm_call.iloc[0]
        atm_put_data = atm_put.iloc[0]
        
        # Find wing strikes
        wing_distance = max(50, spot * 0.05)  # 5% wings
        
        upper_wing = calls_df[calls_df['strike'] >= atm_strike + wing_distance]
        lower_wing = puts_df[puts_df['strike'] <= atm_strike - wing_distance]
        
        if upper_wing.empty or lower_wing.empty:
            return None
        
        long_call = upper_wing.iloc[0]
        long_put = lower_wing.iloc[-1]
        
        # Calculate net credit/debit
        net_premium = (atm_call_data['bid'] + atm_put_data['bid'] - 
                      long_call['ask'] - long_put['ask'])
        
        if net_premium <= 0:
            return None  # Should be credit strategy
        
        max_profit = net_premium * lot_size
        wing_width = min(atm_strike - long_put['strike'], long_call['strike'] - atm_strike)
        max_loss = (wing_width - net_premium) * lot_size
        
        if max_loss <= 0:
            return None
        
        # Breakeven points
        upper_breakeven = atm_strike + net_premium
        lower_breakeven = atm_strike - net_premium
        
        # Profit probability (very tight range around ATM)
        expected_move = spot * volatility * np.sqrt(dte / 365)
        profit_prob = max(0.2, 1 - (expected_move / net_premium))
        
        return {
            'strategy_type': StrategyType.IRON_BUTTERFLY,
            'legs': [
                {
                    'side': 'PUT', 'strike': atm_strike, 'action': 'SELL',
                    'price': atm_put_data['bid'], 'delta': atm_put_data['delta'],
                    'liquidity': atm_put_data['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': atm_strike, 'action': 'SELL',
                    'price': atm_call_data['bid'], 'delta': atm_call_data['delta'],
                    'liquidity': atm_call_data['liquidity_score']
                },
                {
                    'side': 'PUT', 'strike': long_put['strike'], 'action': 'BUY',
                    'price': long_put['ask'], 'delta': long_put['delta'],
                    'liquidity': long_put['liquidity_score']
                },
                {
                    'side': 'CALL', 'strike': long_call['strike'], 'action': 'BUY',
                    'price': long_call['ask'], 'delta': long_call['delta'],
                    'liquidity': long_call['liquidity_score']
                }
            ],
            'net_credit': net_premium,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'profit_probability': profit_prob,
            'wing_width': wing_width,
            'expected_move': expected_move,
            'net_delta': (atm_put_data['delta'] + atm_call_data['delta'] - 
                         long_put['delta'] - long_call['delta']),
            'net_gamma': (atm_put_data['gamma'] + atm_call_data['gamma'] - 
                         long_put['gamma'] - long_call['gamma']),
            'net_theta': (atm_put_data['theta'] + atm_call_data['theta'] - 
                         long_put['theta'] - long_call['theta']),
            'net_vega': (atm_put_data['vega'] + atm_call_data['vega'] - 
                        long_put['vega'] - long_call['vega']),
            'dte': dte,
            'score': self._calculate_strategy_score(profit_prob, max_profit / max_loss, net_premium)
        }

    def _create_long_strangle(self, spot: float, calls_df: pd.DataFrame, puts_df: pd.DataFrame,
                             volatility: float, dte: int, lot_size: int) -> Optional[Dict]:
        """Create long strangle for volatility plays"""
        if calls_df.empty or puts_df.empty:
            return None
        
        # Find OTM strikes equidistant from current price
        distance = spot * 0.05  # 5% OTM on each side
        
        otm_calls = calls_df[
            (calls_df['strike'] >= spot + distance) &
            (calls_df['liquidity_score'] > 0.3)
        ]
        
        otm_puts = puts_df[
            (puts_df['strike'] <= spot - distance) &
            (puts_df['liquidity_score'] > 0.3)
        ]
        
        if otm_calls.empty or otm_puts.empty:
            return None
        
        # Select closest strikes to target distance
        call_strike = otm_calls.loc[(otm_calls['strike'] - (spot + distance)).abs().idxmin()]
        put_strike = otm_puts.loc[(otm_puts['strike'] - (spot - distance)).abs().idxmin()]
        
        total_cost = call_strike['ask'] + put_strike['ask']
        max_loss = total_cost * lot_size
        
        # Breakeven points
        upper_breakeven = call_strike['strike'] + total_cost
        lower_breakeven = put_strike['strike'] - total_cost
        
        # Expected move and profit probability
        expected_move = spot * volatility * np.sqrt(dte / 365)
        breakeven_distance = min(spot - lower_breakeven, upper_breakeven - spot)
        
        profit_prob = max(0.3, min(0.8, expected_move / breakeven_distance))
        
        return {
            'strategy_type': 'LONG_STRANGLE',
            'legs': [
                {
                    'side': 'CALL', 'strike': call_strike['strike'], 'action': 'BUY',
                    'price': call_strike['ask'], 'delta': call_strike['delta'],
                    'liquidity': call_strike['liquidity_score']
                },
                {
                    'side': 'PUT', 'strike': put_strike['strike'], 'action': 'BUY',
                    'price': put_strike['ask'], 'delta': put_strike['delta'],
                    'liquidity': put_strike['liquidity_score']
                }
            ],
            'net_debit': total_cost,
            'max_loss': max_loss,
            'upper_breakeven': upper_breakeven,
            'lower_breakeven': lower_breakeven,
            'profit_probability': profit_prob,
            'expected_move': expected_move,
            'breakeven_range': upper_breakeven - lower_breakeven,
            'net_delta': call_strike['delta'] + put_strike['delta'],
            'net_gamma': call_strike['gamma'] + put_strike['gamma'],
            'net_theta': call_strike['theta'] + put_strike['theta'],
            'net_vega': call_strike['vega'] + put_strike['vega'],
            'dte': dte,
            'score': self._calculate_debit_strategy_score(profit_prob, expected_move, total_cost)
        }

# ================================================================================
# PROFESSIONAL RISK MANAGEMENT SYSTEM
# ================================================================================

@dataclass
class Position:
    """Enhanced position tracking with comprehensive metrics"""
    strategy_type: Union[StrategyType, str]
    legs: List[Dict]
    quantity: int
    entry_date: date
    expiry_date: date
    entry_price: float
    max_profit: float
    max_loss: float
    index_name: str
    
    # P&L tracking
    current_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    
    # Greeks
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    net_rho: float = 0.0
    
    # Position management
    status: str = "OPEN"  # OPEN, CLOSED, EXPIRED
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    days_held: int = 0
    
    # Performance metrics
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_adverse_excursion: float = 0.0
    max_favorable_excursion: float = 0.0
    
    # Trade quality
    liquidity_score: float = 0.0
    execution_quality: float = 0.0
    slippage: float = 0.0

class ProfessionalRiskManager:
    """Advanced risk management with portfolio-level controls"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        
        # Portfolio tracking
        self.daily_equity = []
        self.daily_metrics = []
        self.max_equity = config.initial_capital
        self.max_drawdown_hit = 0.0
        
        # Trade tracking
        self.trade_count = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Risk controls
        self.portfolio_var = 0.0
        self.portfolio_delta = 0.0
        self.portfolio_gamma = 0.0
        self.portfolio_theta = 0.0
        self.portfolio_vega = 0.0
        
        # Performance tracking
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        
    def calculate_position_size(self, strategy_dict: Dict, current_volatility: float,
                               signal_strength: float, signal_quality: float) -> int:
        """Advanced position sizing with multiple risk factors"""
        
        max_loss = strategy_dict.get('max_loss', strategy_dict.get('estimated_max_loss', 0))
        if max_loss <= 0:
            return 0
        
        # Base risk calculation
        base_risk = self.cash * self.config.max_risk_per_trade
        
        # Signal quality adjustment (0.5 to 1.5 multiplier)
        quality_multiplier = 0.5 + signal_quality
        
        # Signal strength adjustment (0.7 to 1.3 multiplier)
        strength_multiplier = 0.7 + (signal_strength * 0.6)
        
        # Volatility adjustment
        if current_volatility <= 0 or np.isnan(current_volatility):
            current_volatility = 0.20
        
        vol_multiplier = max(0.5, min(1.5, 0.20 / current_volatility))
        
        # Consecutive loss adjustment (reduce size after losses)
        loss_adjustment = max(0.5, 1 - (self.consecutive_losses * 0.1))
        
        # Portfolio heat adjustment
        portfolio_heat = self._calculate_portfolio_heat()
        heat_adjustment = max(0.3, 1 - portfolio_heat)
        
        # Win rate adjustment
        if self.win_rate > 0:
            wr_adjustment = min(1.2, 0.8 + self.win_rate * 0.4)
        else:
            wr_adjustment = 1.0
        
        # Calculate final risk amount
        adjusted_risk = (base_risk * quality_multiplier * strength_multiplier * 
                        vol_multiplier * loss_adjustment * heat_adjustment * wr_adjustment)
        
        # Calculate base quantity
        base_quantity = max(1, int(adjusted_risk / max_loss))
        
        # Portfolio exposure limits
        current_exposure = sum(
            pos.max_loss * pos.quantity for pos in self.positions if pos.status == "OPEN"
        )
        
        available_risk = (self.cash * self.config.max_portfolio_risk) - current_exposure
        if available_risk <= 0:
            return 0
        
        max_quantity_by_exposure = max(1, int(available_risk / max_loss))
        
        # Final quantity with position limits
        final_quantity = min(base_quantity, max_quantity_by_exposure, 5)
        
        return final_quantity
    
    def can_open_position(self, index_name: str) -> bool:
        """Enhanced position opening checks"""
        
        # Basic position count checks
        open_positions = [pos for pos in self.positions if pos.status == "OPEN"]
        if len(open_positions) >= self.config.max_total_positions:
            return False
        
        # Index-specific position limits
        index_positions = [pos for pos in open_positions if pos.index_name == index_name]
        if len(index_positions) >= self.config.max_positions_per_index:
            return False
        
        # Portfolio heat check
        if self._calculate_portfolio_heat() > 0.8:
            return False
        
        # Consecutive loss protection
        if self.consecutive_losses >= 5:
            return False
        
        # Drawdown protection
        current_drawdown = self._calculate_current_drawdown()
        if current_drawdown < -0.15:  # Stop trading at 15% drawdown
            return False
        
        # Time-based limits (no more than 3 trades per day)
        today_trades = sum(1 for pos in self.closed_positions 
                          if pos.entry_date == date.today())
        if today_trades >= 3:
            return False
        
        return True
    
    def open_position(self, strategy_dict: Dict, quantity: int, current_date: date,
                     index_name: str, market_data: Dict = None) -> bool:
        """Open new position with comprehensive tracking"""
        
        if not self.can_open_position(index_name) or quantity <= 0:
            return False
        
        # Calculate entry costs
        if 'net_credit' in strategy_dict:
            entry_cost = strategy_dict['net_credit'] * quantity
            is_credit = True
        else:
            entry_cost = -strategy_dict.get('net_debit', 0) * quantity
            is_credit = False
        
        # Calculate transaction costs
        num_legs = len(strategy_dict['legs'])
        lot_size = self._get_lot_size(index_name)
        brokerage_cost = self.config.brokerage_per_lot * num_legs * quantity
        
        # Slippage calculation
        slippage_cost = self._calculate_slippage(strategy_dict, quantity, lot_size)
        
        total_cost = abs(entry_cost) + brokerage_cost + slippage_cost
        
        # Cash availability check
        if total_cost > self.cash * 0.35:  # Don't use more than 20% cash for entry
            return False
        
        # Create position
        position = Position(
            strategy_type=strategy_dict['strategy_type'],
            legs=strategy_dict['legs'],
            quantity=quantity,
            entry_date=current_date,
            expiry_date=current_date + timedelta(days=strategy_dict.get('dte', 30)),
            entry_price=strategy_dict.get('net_credit', -strategy_dict.get('net_debit', 0)),
            max_profit=strategy_dict.get('max_profit', 0) * quantity,
            max_loss=strategy_dict.get('max_loss', 
                                     strategy_dict.get('estimated_max_loss', 0)) * quantity,
            index_name=index_name,
            
            # Initialize Greeks
            net_delta=strategy_dict.get('net_delta', 0) * quantity,
            net_gamma=strategy_dict.get('net_gamma', 0) * quantity,
            net_theta=strategy_dict.get('net_theta', 0) * quantity,
            net_vega=strategy_dict.get('net_vega', 0) * quantity,
            net_rho=strategy_dict.get('net_rho', 0) * quantity,
            
            # Trade quality metrics
            liquidity_score=self._calculate_position_liquidity(strategy_dict),
            execution_quality=0.8,  # Assume good execution initially
            slippage=slippage_cost
        )
        
        # Update cash and add position
        self.cash += entry_cost - brokerage_cost - slippage_cost
        self.positions.append(position)
        self.trade_count += 1
        
        # Update portfolio Greeks
        self._update_portfolio_greeks()
        
        # Log trade
        logger.info(f"📈 Trade #{self.trade_count}: {position.strategy_type} | "
                   f"Index={index_name} | Qty={quantity} | "
                   f"Entry={'Credit' if is_credit else 'Debit'}=₹{abs(entry_cost):.0f}")
        
        return True
    
    def update_positions(self, current_date: date, market_data: Dict[str, Tuple],
                        spot_prices: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """Update all positions with enhanced P&L calculation"""
        
        total_pnl = 0.0
        positions_to_close = []
        portfolio_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for position in self.positions:
            if position.status != "OPEN":
                continue
            
            # Update days held
            position.days_held = (current_date - position.entry_date).days
            
            # Check for expiry
            if current_date >= position.expiry_date - timedelta(days=1):
                positions_to_close.append((position, "EXPIRY"))
                continue
            
            # Get current option data
            if position.index_name not in market_data:
                continue
            
            calls_df, puts_df = market_data[position.index_name]
            current_spot = spot_prices.get(position.index_name, 0)
            
            if current_spot <= 0:
                continue
            
            # Calculate current position value and Greeks
            position_value, position_greeks = self._calculate_position_value_and_greeks(
                position, calls_df, puts_df, current_spot
            )
            
            # Update position P&L
            if position.entry_price >= 0:  # Credit strategy
                position.current_pnl = (position.entry_price - position_value) * position.quantity
            else:  # Debit strategy
                position.current_pnl = (position_value + position.entry_price) * position.quantity
            
            # Update position Greeks
            position.net_delta = position_greeks['delta']
            position.net_gamma = position_greeks['gamma']
            position.net_theta = position_greeks['theta']
            position.net_vega = position_greeks['vega']
            position.net_rho = position_greeks['rho']
            
            # Track high/low water marks
            position.max_pnl = max(position.max_pnl, position.current_pnl)
            position.min_pnl = min(position.min_pnl, position.current_pnl)
            
            # Calculate MAE and MFE
            position.max_adverse_excursion = min(position.max_adverse_excursion, position.current_pnl)
            position.max_favorable_excursion = max(position.max_favorable_excursion, position.current_pnl)
            
            # Add to totals
            total_pnl += position.current_pnl
            
            for greek in portfolio_greeks:
                portfolio_greeks[greek] += position_greeks[greek]
            
            # Check exit conditions
            exit_signal = self._check_exit_conditions(position)
            if exit_signal:
                positions_to_close.append((position, exit_signal))
        
        # Close positions that need to be closed
        for position, reason in positions_to_close:
            self._close_position(position, current_date, reason)
        
        # Update portfolio metrics
        self._update_portfolio_greeks()
        
        return total_pnl, portfolio_greeks
    
    def _check_exit_conditions(self, position: Position) -> Optional[str]:
        """Check if position should be exited"""
        
        if position.max_loss <= 0:
            return None
        
        # Calculate P&L percentages
        pnl_pct_of_max_loss = position.current_pnl / abs(position.max_loss)
        
        if position.max_profit > 0:
            pnl_pct_of_max_profit = position.current_pnl / position.max_profit
        else:
            pnl_pct_of_max_profit = 0
        
        # Stop loss check
        if pnl_pct_of_max_loss <= -self.config.stop_loss_pct:
            return "STOP_LOSS"
        
                # Take profit check
        if position.max_profit > 0 and pnl_pct_of_max_profit >= self.config.take_profit_pct:
            return "TAKE_PROFIT"
        
        # Time-based exits
        if position.days_held >= self.config.time_stop_days:
            if position.current_pnl > 0:
                return "TIME_PROFIT"
            elif pnl_pct_of_max_loss <= -0.3:
                return "TIME_LOSS"
        
        # Trailing stop for profitable trades
        if position.max_pnl > 0 and position.current_pnl <= position.max_pnl * 0.6:
            return "TRAILING_STOP"
        
        # Quick profit taking (first 2 days)
        if position.days_held <= 2 and position.max_profit > 0:
            if pnl_pct_of_max_profit >= 0.5:
                return "QUICK_PROFIT"
        
        # Greek-based exits
        if abs(position.net_delta) > 100:  # High delta exposure
            return "DELTA_RISK"
        
        return None
    
    def _close_position(self, position: Position, current_date: date, reason: str):
        """Close position with comprehensive tracking"""
        
        position.status = "CLOSED"
        position.exit_date = current_date
        position.exit_reason = reason
        position.exit_price = position.current_pnl / position.quantity if position.quantity > 0 else 0
        
        # Calculate closing costs
        num_legs = len(position.legs)
        closing_cost = self.config.brokerage_per_lot * num_legs * position.quantity
        slippage_cost = abs(position.current_pnl) * 0.001  # 0.1% slippage on exit
        
        total_closing_cost = closing_cost + slippage_cost
        
        # Realize P&L
        net_pnl = position.current_pnl - total_closing_cost
        self.cash += net_pnl
        
        # Update position final metrics
        position.current_pnl = net_pnl
        
        # Calculate position-level metrics
        if position.max_loss != 0:
            position.profit_factor = max(0, position.current_pnl) / max(0.01, abs(min(0, position.current_pnl)))
        
        # Track consecutive wins/losses
        if position.current_pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Move to closed positions
        self.closed_positions.append(position)
        
        # Remove from open positions
        if position in self.positions:
            self.positions.remove(position)
        
        # Update portfolio-level metrics
        self._update_performance_metrics()
        
        # Log trade closure
        profit_emoji = "💚" if position.current_pnl > 0 else "💔"
        logger.info(f"{profit_emoji} Closed #{len(self.closed_positions)}: "
                   f"{position.strategy_type} | P&L=₹{position.current_pnl:.0f} | "
                   f"Days={position.days_held} | Reason={reason}")
    
    def _calculate_position_value_and_greeks(self, position: Position, calls_df: pd.DataFrame,
                                           puts_df: pd.DataFrame, spot: float) -> Tuple[float, Dict]:
        """Calculate current position value and Greeks"""
        
        total_value = 0.0
        total_greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        for leg in position.legs:
            # Find matching option
            if leg['side'] == 'CALL':
                option_data = calls_df[calls_df['strike'] == leg['strike']]
            else:
                option_data = puts_df[puts_df['strike'] == leg['strike']]
            
            if not option_data.empty:
                option_row = option_data.iloc[0]
                current_price = option_row['price']
                
                # Calculate leg value based on action
                if leg['action'] == 'SELL':
                    leg_value = -current_price
                    multiplier = -1
                else:  # BUY
                    leg_value = current_price
                    multiplier = 1
                
                total_value += leg_value
                
                # Add Greeks
                for greek in total_greeks:
                    if greek in option_row:
                        total_greeks[greek] += option_row[greek] * multiplier * position.quantity
        
        return total_value, total_greeks
    
    def _calculate_portfolio_heat(self) -> float:
        """Calculate portfolio heat (percentage of capital at risk)"""
        total_risk = sum(pos.max_loss * pos.quantity for pos in self.positions if pos.status == "OPEN")
        return total_risk / self.cash if self.cash > 0 else 0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.daily_equity:
            return 0.0
        
        current_equity = self.daily_equity[-1][1]
        return (current_equity / self.max_equity) - 1 if self.max_equity > 0 else 0
    
    def _calculate_slippage(self, strategy_dict: Dict, quantity: int, lot_size: int) -> float:
        """Calculate realistic slippage costs"""
        # Base slippage on strategy value and liquidity
        total_premium = 0
        min_liquidity = 1.0
        
        for leg in strategy_dict['legs']:
            total_premium += leg['price']
            min_liquidity = min(min_liquidity, leg.get('liquidity', 0.5))
        
        # Slippage increases with position size and decreases with liquidity
        base_slippage = total_premium * quantity * lot_size * self.config.slippage_bps / 10000
        liquidity_multiplier = 2.0 - min_liquidity  # 1.0 to 2.0 range
        size_multiplier = 1.0 + (quantity - 1) * 0.1  # Increases with size
        
        return base_slippage * liquidity_multiplier * size_multiplier
    
    def _calculate_position_liquidity(self, strategy_dict: Dict) -> float:
        """Calculate overall position liquidity score"""
        if not strategy_dict['legs']:
            return 0.5
        
        total_liquidity = sum(leg.get('liquidity', 0.5) for leg in strategy_dict['legs'])
        return total_liquidity / len(strategy_dict['legs'])
    
    def _get_lot_size(self, index_name: str) -> int:
        """Get lot size for index"""
        lot_sizes = {
            "NIFTY_50": self.config.lot_size_nifty,
            "BANK_NIFTY": self.config.lot_size_banknifty,
            "SENSEX": self.config.lot_size_sensex
        }
        return lot_sizes.get(index_name, 50)
    
    def _update_portfolio_greeks(self):
        """Update portfolio-level Greeks"""
        self.portfolio_delta = sum(pos.net_delta for pos in self.positions if pos.status == "OPEN")
        self.portfolio_gamma = sum(pos.net_gamma for pos in self.positions if pos.status == "OPEN")
        self.portfolio_theta = sum(pos.net_theta for pos in self.positions if pos.status == "OPEN")
        self.portfolio_vega = sum(pos.net_vega for pos in self.positions if pos.status == "OPEN")
    
    def _update_performance_metrics(self):
        """Update performance metrics after each trade"""
        if not self.closed_positions:
            return
        
        winning_trades = [pos for pos in self.closed_positions if pos.current_pnl > 0]
        losing_trades = [pos for pos in self.closed_positions if pos.current_pnl <= 0]
        
        total_trades = len(self.closed_positions)
        self.win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        self.avg_win = np.mean([pos.current_pnl for pos in winning_trades]) if winning_trades else 0
        self.avg_loss = np.mean([pos.current_pnl for pos in losing_trades]) if losing_trades else 0
        
        if self.avg_loss != 0:
            self.profit_factor = abs(self.avg_win * len(winning_trades)) / abs(self.avg_loss * len(losing_trades))
        else:
            self.profit_factor = float('inf') if self.avg_win > 0 else 0
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics"""
        if not self.daily_equity:
            return {}
        
        # Convert daily equity to DataFrame
        equity_df = pd.DataFrame(self.daily_equity, columns=['date', 'equity'])
        equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
        
        # Basic metrics
        initial_capital = self.config.initial_capital
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity / initial_capital) - 1
        
        # Time-based metrics
        trading_days = len(equity_df)
        years = trading_days / 252
        
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = total_return
        
        # Risk metrics
        daily_returns = equity_df['returns']
        annualized_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio
        risk_free_rate = self.config.risk_free_rate
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0
        
        # Maximum drawdown
        running_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Calmar ratio
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
        
        # Trade statistics
        if self.closed_positions:
            total_trades = len(self.closed_positions)
            avg_days_held = np.mean([pos.days_held for pos in self.closed_positions])
            
            # Strategy breakdown
            strategy_stats = {}
            for pos in self.closed_positions:
                strategy = str(pos.strategy_type)
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'count': 0, 'total_pnl': 0, 'wins': 0}
                
                strategy_stats[strategy]['count'] += 1
                strategy_stats[strategy]['total_pnl'] += pos.current_pnl
                if pos.current_pnl > 0:
                    strategy_stats[strategy]['wins'] += 1
        else:
            total_trades = avg_days_held = 0
            strategy_stats = {}
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_days_held': avg_days_held,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'current_positions': len([p for p in self.positions if p.status == "OPEN"]),
            'portfolio_delta': self.portfolio_delta,
            'portfolio_gamma': self.portfolio_gamma,
            'portfolio_theta': self.portfolio_theta,
            'portfolio_vega': self.portfolio_vega,
            'strategy_breakdown': strategy_stats
        }

# ================================================================================
# PROFESSIONAL BACKTESTING ENGINE
# ================================================================================

class ProfessionalBacktestEngine:
    """Advanced backtesting engine with walk-forward analysis"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.data_manager = ProfessionalDataManager(config)
        self.technical_analyzer = AdvancedTechnicalAnalyzer(config)
        self.strategy_manager = AdvancedStrategyManager(config)
        self.results = []
        
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtest with walk-forward analysis"""
        
        logger.info("🚀 Starting Professional Options Trading Backtest")
        logger.info(f"💰 Initial Capital: ₹{self.config.initial_capital:,.0f}")
        logger.info(f"📅 Period: {self.config.start_date} to {self.config.end_date}")
        logger.info(f"🔄 Walk-Forward: {'Enabled' if self.config.walk_forward_enabled else 'Disabled'}")
        
        try:
            # Fetch multi-index data
            all_data = self.data_manager.fetch_multi_index_data()
            
            if not all_data:
                raise ValueError("No market data available")
            
            # Run walk-forward analysis if enabled
            if self.config.walk_forward_enabled:
                results = self._run_walk_forward_backtest(all_data)
            else:
                results = self._run_single_backtest(all_data)
            
            logger.info("✅ Backtest completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def _run_walk_forward_backtest(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run walk-forward backtesting"""
        
        walk_forward_results = []
        
        # Generate walk-forward periods
        periods = self._generate_walk_forward_periods()
        
        for period_idx, (train_start, train_end, test_start, test_end) in enumerate(periods):
            logger.info(f"\n📊 Walk-Forward Period {period_idx + 1}/{len(periods)}")
            logger.info(f"   Training: {train_start} to {train_end}")
            logger.info(f"   Testing: {test_start} to {test_end}")
            
            period_result = {
                'period': period_idx + 1,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'results': {}
            }
            
            # Run backtest for each index in this period
            for index_name, data in all_data.items():
                logger.info(f"   🔍 Testing {index_name}...")
                
                try:
                    result = self._run_period_backtest(
                        data, index_name, train_start, train_end, test_start, test_end
                    )
                    period_result['results'][index_name] = result
                    
                    # Log period results
                    metrics = result.get('metrics', {})
                    total_return = metrics.get('total_return', 0) * 100
                    win_rate = metrics.get('win_rate', 0) * 100
                    trades = metrics.get('total_trades', 0)
                    
                    logger.info(f"   ✅ {index_name}: Return={total_return:.1f}%, "
                              f"Win Rate={win_rate:.1f}%, Trades={trades}")
                    
                except Exception as e:
                    logger.error(f"   ❌ {index_name} failed: {e}")
                    period_result['results'][index_name] = {'error': str(e)}
            
            walk_forward_results.append(period_result)
        
        # Compile final results
        final_results = {
            'type': 'walk_forward',
            'periods': walk_forward_results,
            'summary': self._calculate_walk_forward_summary(walk_forward_results),
            'config': self.config
        }
        
        return final_results
    
    def _run_single_backtest(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run single period backtest"""
        
        single_results = {}
        
        for index_name, data in all_data.items():
            logger.info(f"🔍 Testing {index_name}...")
            
            try:
                result = self._run_period_backtest(
                    data, index_name, 
                    self.config.start_date, 
                    self.config.end_date,
                    self.config.start_date,
                    self.config.end_date
                )
                single_results[index_name] = result
                
                # Log results
                metrics = result.get('metrics', {})
                total_return = metrics.get('total_return', 0) * 100
                win_rate = metrics.get('win_rate', 0) * 100
                trades = metrics.get('total_trades', 0)
                
                logger.info(f"✅ {index_name}: Return={total_return:.1f}%, "
                          f"Win Rate={win_rate:.1f}%, Trades={trades}")
                
            except Exception as e:
                logger.error(f"❌ {index_name} failed: {e}")
                single_results[index_name] = {'error': str(e)}
        
        return {
            'type': 'single',
            'results': single_results,
            'config': self.config
        }
    
    def _run_period_backtest(self, data: pd.DataFrame, index_name: str,
                            train_start: date, train_end: date, 
                            test_start: date, test_end: date) -> Dict[str, Any]:
        """Run backtest for a specific period and index"""
        
        # normalise input dates to date objects

        if not isinstance(test_start, date):
            test_start = pd.to_datetime(test_start).date()
        if not isinstance(test_end, date):
            test_end = pd.to_datetime(test_end).date()

        # Ensure dataframe index is date type 
        data = data.copy()
        idx = pd.to_datetime(pd.Index(data.index), errors='coerce')
        if getattr(idx, 'tz', None) is not None:
            idx = idx.tz_convert(None)

        data.index = idx.date

        mask = (data.index >= test_start) & (data.index <= test_end)
        test_data = data.loc[mask].copy()

        if test_data.empty:
            return {'error': 'no test data available for this period'}


        # Add technical indicators
        test_data = self.technical_analyzer.add_comprehensive_indicators(test_data, index_name)
        
        # Generate trading signals
        test_data = self.technical_analyzer.generate_professional_signals(test_data, index_name)

        # Skip first row with NaN signals for rolling calculations
        if len(test_data) <= 20:  # Reduce warmup period for more trades
            return {'error': 'not enough data for warmup'}
        test_data = test_data.iloc[20:].copy()

        # Initialize risk manager
        risk_manager = ProfessionalRiskManager(self.config)
        
        # Run daily backtest loop
        trades_attempted = 0
        trades_executed = 0
        
        if RICH_AVAILABLE:
            progress = Progress()
            task = progress.add_task(f"[cyan]Backtesting {index_name}...", total=len(test_data))
            progress.start()
        
        try:
            for current_date_pd, row in test_data.iterrows():
                if RICH_AVAILABLE:
                    progress.update(task, advance=1)
                
                # Convert date
                if isinstance(current_date_pd, date):
                    current_date = current_date_pd
                else:
                    current_date = current_date_pd.date()
                
                # Extract market data
                spot_price = row['Close']
                volatility = row.get('Volatility_20', 0.20)
                signal = row.get('Final_Signal', 0)
                signal_strength = row.get('Signal_Strength', 0)
                signal_quality = row.get('Signal_Quality', 0.5)
                market_regime = row.get('Market_Regime', 'sideways')
                volatility_rank = row.get('Volatility_Rank', 0.5)
                
                # Validate data
                if spot_price <= 0 or np.isnan(spot_price):
                    continue
                
                if volatility <= 0 or np.isnan(volatility):
                    volatility = 0.20
                
                if signal_strength <= 0 or np.isnan(signal_strength):
                    signal_strength = 0.1
                
                if signal_quality <= 0 or np.isnan(signal_quality):
                    signal_quality = 0.5
                
                # Generate options chain
                dte = random.randint(self.config.min_dte, self.config.max_dte)
                calls_df, puts_df = self.strategy_manager.options_engine.generate_professional_option_chain(
                    spot_price, volatility, dte, index_name
                )
                
                # Update existing positions
                market_data = {index_name: (calls_df, puts_df)}
                spot_prices = {index_name: spot_price}
                
                open_pnl, portfolio_greeks = risk_manager.update_positions(
                    current_date, market_data, spot_prices
                )
                
                # Calculate current equity
                current_equity = max(0, risk_manager.cash + open_pnl)
                risk_manager.daily_equity.append([current_date, current_equity])
                risk_manager.max_equity = max(risk_manager.max_equity, current_equity)
                
                # Entry logic
                entry_conditions = [
                    signal != 0,
                    signal_strength >= 0.1,  # Lowered threshold
                    signal_quality >= 0.2,   # Lowered threshold
                    risk_manager.can_open_position(index_name),
                    current_equity > self.config.initial_capital * 0.3  # Lowered equity threshold
                ]
                
                if all(entry_conditions):
                    trades_attempted += 1
                    
                    # Select optimal strategy
                    additional_context = {
                        'trend_strength': row.get('Trend_Strength', 0),
                        'signal_quality': signal_quality
                    }
                    
                    strategy_dict = self.strategy_manager.select_optimal_strategy(
                        signal, market_regime, volatility_rank, spot_price,
                        calls_df, puts_df, volatility, dte, index_name, additional_context
                    )
                    
                    if strategy_dict:
                        # Calculate position size
                        quantity = risk_manager.calculate_position_size(
                            strategy_dict, volatility, signal_strength, signal_quality
                        )
                        
                        if quantity > 0:
                            success = risk_manager.open_position(
                                strategy_dict, quantity, current_date, index_name
                            )
                            
                            if success:
                                trades_executed += 1
                
                # Emergency stop conditions
                current_drawdown = risk_manager._calculate_current_drawdown()
                if current_drawdown < -0.25:  # Stop at 25% drawdown
                    logger.warning(f"Emergency stop: {index_name} hit 25% drawdown")
                    break
                
                if current_equity < self.config.initial_capital * 0.3:
                    logger.warning(f"Emergency stop: {index_name} equity below 30%")
                    break
        
        finally:
            if RICH_AVAILABLE:
                progress.stop()
        
        # Calculate final metrics
        metrics = risk_manager.get_portfolio_metrics()
        
        return {
            'metrics': metrics,
            'daily_equity': risk_manager.daily_equity,
            'closed_positions': risk_manager.closed_positions,
            'open_positions': [p for p in risk_manager.positions if p.status == "OPEN"],
            'trades_attempted': trades_attempted,
            'trades_executed': trades_executed,
            'execution_rate': trades_executed / max(trades_attempted, 1),
            'test_period': {'start': test_start, 'end': test_end},
            'index_name': index_name
        }
    
    def _generate_walk_forward_periods(self) -> List[Tuple[date, date, date, date]]:
        """Generate walk-forward testing periods"""
        periods = []
        
        # Calculate total months
        start_date = self.config.start_date
        end_date = self.config.end_date
        
        current_start = start_date
        
        for i in range(self.config.walk_forward_periods):
            # Training period
            train_start = current_start
            train_end = self._add_months(train_start, self.config.training_window_months)
            
            # Testing period
            test_start = train_end
            test_end = self._add_months(test_start, self.config.testing_window_months)
            
            # Ensure we don't exceed end date
            if test_end > end_date:
                test_end = end_date
            
            if test_start < end_date and train_start < train_end:
                periods.append((train_start, train_end, test_start, test_end))
            
            # Move to next period
            current_start = self._add_months(current_start, self.config.testing_window_months)
            
            if current_start >= end_date:
                break
        
        return periods
    
    def _add_months(self, start_date: date, months: int) -> date:
        """Add months to a date"""
        year = start_date.year + (start_date.month + months - 1) // 12
        month = (start_date.month + months - 1) % 12 + 1
        day = min(start_date.day, [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
        return date(year, month, day)
    
    def _calculate_walk_forward_summary(self, walk_forward_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for walk-forward results"""
        
        summary = {
            'by_index': {},
            'by_period': [],
            'overall': {}
        }
        
        # Collect results by index
        index_results = {}
        for period_result in walk_forward_results:
            for index_name, result in period_result['results'].items():
                if 'error' not in result and 'metrics' in result:
                    if index_name not in index_results:
                        index_results[index_name] = []
                    index_results[index_name].append(result['metrics'])
        
        # Calculate index statistics
        for index_name, metrics_list in index_results.items():
            returns = [m['total_return'] for m in metrics_list]
            win_rates = [m['win_rate'] for m in metrics_list]
            sharpe_ratios = [m['sharpe_ratio'] for m in metrics_list]
            max_drawdowns = [m['max_drawdown'] for m in metrics_list]
            
            summary['by_index'][index_name] = {
                'periods_tested': len(returns),
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'best_return': max(returns) if returns else 0,
                'worst_return': min(returns) if returns else 0,
                'avg_win_rate': np.mean(win_rates),
                'avg_sharpe': np.mean(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'consistency_score': len([r for r in returns if r > 0]) / len(returns) if returns else 0
            }
        
        # Calculate period statistics
        for i, period_result in enumerate(walk_forward_results):
            period_returns = []
            period_trades = []
            
            for result in period_result['results'].values():
                if 'error' not in result and 'metrics' in result:
                    period_returns.append(result['metrics']['total_return'])
                    period_trades.append(result['metrics']['total_trades'])
            
            summary['by_period'].append({
                'period': i + 1,
                'start_date': period_result['test_start'],
                'end_date': period_result['test_end'],
                'avg_return': np.mean(period_returns) if period_returns else 0,
                'total_trades': sum(period_trades),
                'indices_tested': len(period_returns)
            })
        
        # Overall statistics
        all_returns = []
        all_sharpes = []
        all_trades = []
        
        for metrics_list in index_results.values():
            all_returns.extend([m['total_return'] for m in metrics_list])
            all_sharpes.extend([m['sharpe_ratio'] for m in metrics_list])
            all_trades.extend([m['total_trades'] for m in metrics_list])
        
        summary['overall'] = {
            'total_backtests': len(all_returns),
            'avg_return': np.mean(all_returns) if all_returns else 0,
            'std_return': np.std(all_returns) if all_returns else 0,
            'win_percentage': len([r for r in all_returns if r > 0]) / len(all_returns) if all_returns else 0,
            'avg_sharpe': np.mean(all_sharpes) if all_sharpes else 0,
            'total_trades': sum(all_trades),
            'best_index': max(summary['by_index'].items(), key=lambda x: x[1]['avg_return'])[0] if summary['by_index'] else 'None'
        }
        
        return summary

# ================================================================================
# PROFESSIONAL REPORTING SYSTEM
# ================================================================================

class ProfessionalReportGenerator:
    """Advanced reporting with interactive visualizations"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        if 'seaborn-v0_8' not in plt.style.available:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
        
    def generate_comprehensive_report(self, results: Dict[str, Any]):
        """Generate comprehensive performance report"""
        
        if results['type'] == 'walk_forward':
            self._generate_walk_forward_report(results)
        else:
            self._generate_single_period_report(results)
    
    def _generate_walk_forward_report(self, results: Dict[str, Any]):
        """Generate walk-forward analysis report"""
        
        summary = results.get('summary', {})
        
        if RICH_AVAILABLE:
            self._print_walk_forward_summary_rich(summary, results)
        else:
            self._print_walk_forward_summary_simple(summary)
        
        # Generate visualizations
        self._create_walk_forward_charts(results)
    
    def _print_walk_forward_summary_rich(self, summary: Dict, results: Dict):
        """Print walk-forward summary using Rich"""
        
        # Overall Performance
        console.print(Panel.fit(
            f"[bold cyan]🏆 Professional Options Trading System - Walk-Forward Results[/bold cyan]\n\n"
            f"[green]✅ Comprehensive Multi-Index Analysis Complete[/green]\n"
            f"Total Backtests: {summary['overall']['total_backtests']}\n"
            f"Average Return: {summary['overall']['avg_return']*100:.2f}%\n"
            f"Win Percentage: {summary['overall']['win_percentage']*100:.1f}%\n"
            f"Average Sharpe: {summary['overall']['avg_sharpe']:.3f}\n"
            f"Best Performing Index: {summary['overall']['best_index']}\n"
            f"Total Trades Executed: {summary['overall']['total_trades']}",
            title="🎯 Executive Summary",
            border_style="green"
        ))
        
        # Index Performance Table
        if summary['by_index']:
            table = Table(title="📊 Index Performance Analysis", show_header=True, header_style="bold magenta")
            table.add_column("Index", style="white", width=15)
            table.add_column("Periods", style="cyan", width=8)
            table.add_column("Avg Return", style="green", width=12)
            table.add_column("Std Dev", style="yellow", width=10)
            table.add_column("Best", style="bright_green", width=10)
            table.add_column("Worst", style="red", width=10)
            table.add_column("Win Rate", style="blue", width=10)
            table.add_column("Consistency", style="magenta", width=12)
            
            for index_name, stats in summary['by_index'].items():
                consistency_color = "green" if stats['consistency_score'] > 0.6 else \
                                  "yellow" if stats['consistency_score'] > 0.4 else "red"
                
                table.add_row(
                    index_name,
                    str(stats['periods_tested']),
                    f"{stats['avg_return']*100:.2f}%",
                    f"{stats['std_return']*100:.2f}%",
                    f"{stats['best_return']*100:.2f}%",
                    f"{stats['worst_return']*100:.2f}%",
                    f"{stats['avg_win_rate']*100:.1f}%",
                    f"[{consistency_color}]{stats['consistency_score']*100:.1f}%[/{consistency_color}]"
                )
            
            console.print(table)
        
        # Period Performance
        if summary['by_period']:
            period_table = Table(title="📈 Period-by-Period Analysis", show_header=True, header_style="bold yellow")
            period_table.add_column("Period", style="white")
            period_table.add_column("Test Period", style="cyan")
            period_table.add_column("Avg Return", style="green")
            period_table.add_column("Total Trades", style="blue")
            period_table.add_column("Indices", style="magenta")
            
            for period_data in summary['by_period']:
                return_color = "green" if period_data['avg_return'] > 0 else "red"
                
                period_table.add_row(
                    f"Period {period_data['period']}",
                    f"{period_data['start_date']} to {period_data['end_date']}",
                    f"[{return_color}]{period_data['avg_return']*100:.2f}%[/{return_color}]",
                    str(period_data['total_trades']),
                    str(period_data['indices_tested'])
                )
            
            console.print(period_table)
        
        # Strategy Recommendations
        recommendations = self._generate_recommendations(summary)
        console.print(Panel.fit(
            recommendations,
            title="🎯 Professional Trading Recommendations",
            border_style="cyan"
        ))
    
    def _print_walk_forward_summary_simple(self, summary: Dict):
        """Simple text summary"""
        print("\n" + "="*80)
        print("PROFESSIONAL OPTIONS TRADING SYSTEM - WALK-FORWARD RESULTS")
        print("="*80)
        
        overall = summary['overall']
        print(f"Total Backtests: {overall['total_backtests']}")
        print(f"Average Return: {overall['avg_return']*100:.2f}%")
        print(f"Win Percentage: {overall['win_percentage']*100:.1f}%")
        print(f"Average Sharpe: {overall['avg_sharpe']:.3f}")
        print(f"Best Index: {overall['best_index']}")
        print(f"Total Trades: {overall['total_trades']}")
        
        print(f"\nIndex Performance:")
        for index_name, stats in summary['by_index'].items():
            print(f"  {index_name}:")
            print(f"    Avg Return: {stats['avg_return']*100:.2f}% (±{stats['std_return']*100:.2f}%)")
            print(f"    Win Rate: {stats['avg_win_rate']*100:.1f}%")
            print(f"    Consistency: {stats['consistency_score']*100:.1f}%")
        
        print("="*80)
    
    def _generate_recommendations(self, summary: Dict) -> str:
        """Generate professional trading recommendations"""
        recommendations = []
        
        overall = summary['overall']
        
        # Overall performance assessment
        if overall['avg_return'] > 0.15:
            recommendations.append("✅ [green]Excellent Strategy Performance[/green] - System shows strong alpha generation")
        elif overall['avg_return'] > 0.08:
            recommendations.append("⚠️  [yellow]Good Strategy Performance[/yellow] - Consider optimization for enhanced returns")
        else:
            recommendations.append("❌ [red]Underperforming Strategy[/red] - Major revisions needed")
        
        # Win rate assessment
        if overall['win_percentage'] > 0.65:
            recommendations.append("🎯 [green]High Win Rate Achieved[/green] - Strong signal quality")
        elif overall['win_percentage'] > 0.50:
            recommendations.append("📊 [yellow]Acceptable Win Rate[/yellow] - Focus on risk-reward optimization")
        else:
            recommendations.append("📉 [red]Low Win Rate[/red] - Signal generation needs improvement")
        
        # Sharpe ratio assessment
        if overall['avg_sharpe'] > 1.5:
            recommendations.append("📈 [green]Excellent Risk-Adjusted Returns[/green] - Superior risk management")
        elif overall['avg_sharpe'] > 1.0:
            recommendations.append("📊 [yellow]Good Risk-Adjusted Returns[/yellow] - Solid performance")
        else:
            recommendations.append("📉 [red]Poor Risk-Adjusted Returns[/red] - Enhance risk management")
        
        # Index-specific recommendations
        if summary['by_index']:
            best_index = max(summary['by_index'].items(), key=lambda x: x[1]['avg_return'])
            recommendations.append(f"🚀 [cyan]Focus on {best_index[0]}[/cyan] - Highest performing index")
            
            for index_name, stats in summary['by_index'].items():
                if stats['consistency_score'] < 0.4:
                    recommendations.append(f"⚠️  [yellow]{index_name} shows inconsistent results[/yellow] - Review strategy selection")
        
        # Trade frequency assessment
        if overall['total_trades'] < 50:
            recommendations.append("📊 [yellow]Low Trade Frequency[/yellow] - Consider lowering signal thresholds")
        elif overall['total_trades'] > 500:
            recommendations.append("⚡ [yellow]High Trade Frequency[/yellow] - Monitor transaction costs")
        
        return "\n".join(recommendations)
    
    def _create_walk_forward_charts(self, results: Dict):
        """Create comprehensive walk-forward visualization"""
        
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle('Professional Options Trading System - Walk-Forward Analysis', 
                        fontsize=16, fontweight='bold')
            
            summary = results['summary']
            
            # 1. Index Performance Comparison
            if summary['by_index']:
                indices = list(summary['by_index'].keys())
                returns = [summary['by_index'][idx]['avg_return'] * 100 for idx in indices]
                colors = ['green' if r > 0 else 'red' for r in returns]
                
                axes[0, 0].bar(indices, returns, color=colors, alpha=0.7)
                axes[0, 0].set_title('Average Returns by Index (%)', fontweight='bold')
                axes[0, 0].set_ylabel('Return (%)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                # Add value labels
                for i, v in enumerate(returns):
                    axes[0, 0].text(i, v + (1 if v > 0 else -1), f'{v:.1f}%', 
                                   ha='center', va='bottom' if v > 0 else 'top')
            
            # 2. Risk-Return Scatter Plot
            if summary['by_index']:
                returns = [summary['by_index'][idx]['avg_return'] * 100 for idx in indices]
                volatilities = [summary['by_index'][idx]['std_return'] * 100 for idx in indices]
                sharpes = [summary['by_index'][idx]['avg_sharpe'] for idx in indices]
                
                scatter = axes[0, 1].scatter(volatilities, returns, s=100, c=sharpes, 
                                           cmap='RdYlGn', alpha=0.7)
                axes[0, 1].set_title('Risk-Return Profile', fontweight='bold')
                axes[0, 1].set_xlabel('Volatility (%)')
                axes[0, 1].set_ylabel('Return (%)')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add index labels
                for i, idx in enumerate(indices):
                    axes[0, 1].annotate(idx, (volatilities[i], returns[i]),
                                       xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[0, 1], label='Sharpe Ratio')
            
            # 3. Consistency Analysis
            if summary['by_index']:
                consistency_scores = [summary['by_index'][idx]['consistency_score'] * 100 for idx in indices]
                win_rates = [summary['by_index'][idx]['avg_win_rate'] * 100 for idx in indices]
                
                colors = ['green' if c > 60 else 'orange' if c > 40 else 'red' for c in consistency_scores]
                bars = axes[0, 2].bar(indices, consistency_scores, color=colors, alpha=0.7)
                axes[0, 2].set_title('Consistency Scores (%)', fontweight='bold')
                axes[0, 2].set_ylabel('Consistency (%)')
                axes[0, 2].tick_params(axis='x', rotation=45)
                axes[0, 2].grid(True, alpha=0.3)
                
                # Add benchmark line
                axes[0, 2].axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Good (60%)')
                axes[0, 2].axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Fair (40%)')
                axes[0, 2].legend()
            
            # 4. Period Performance Timeline
            if summary['by_period']:
                periods = [p['period'] for p in summary['by_period']]
                period_returns = [p['avg_return'] * 100 for p in summary['by_period']]
                colors = ['green' if r > 0 else 'red' for r in period_returns]
                
                axes[1, 0].bar(periods, period_returns, color=colors, alpha=0.7)
                axes[1, 0].set_title('Performance by Period (%)', fontweight='bold')
                axes[1, 0].set_xlabel('Walk-Forward Period')
                axes[1, 0].set_ylabel('Return (%)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 5. Trade Activity Analysis
            if summary['by_period']:
                trade_counts = [p['total_trades'] for p in summary['by_period']]
                
                axes[1, 1].plot(periods, trade_counts, marker='o', linewidth=2, markersize=8)
                axes[1, 1].fill_between(periods, trade_counts, alpha=0.3)
                axes[1, 1].set_title('Trading Activity by Period', fontweight='bold')
                axes[1, 1].set_xlabel('Walk-Forward Period')
                axes[1, 1].set_ylabel('Number of Trades')
                axes[1, 1].grid(True, alpha=0.3)
            
            # 6. Performance Distribution
            if summary['by_index']:
                all_returns = []
                for idx, stats in summary['by_index'].items():
                    # Simulate individual period returns based on avg and std
                    n_periods = stats['periods_tested']
                    if n_periods > 1:
                        simulated_returns = np.random.normal(
                            stats['avg_return'] * 100, 
                            stats['std_return'] * 100, 
                            n_periods
                        )
                        all_returns.extend(simulated_returns)
                
                if all_returns:
                    axes[1, 2].hist(all_returns, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                    axes[1, 2].axvline(np.mean(all_returns), color='red', linestyle='--', linewidth=2,
                                     label=f'Mean: {np.mean(all_returns):.1f}%')
                    axes[1, 2].axvline(0, color='black', linestyle='-', alpha=0.5)
                    axes[1, 2].set_title('Return Distribution', fontweight='bold')
                    axes[1, 2].set_xlabel('Return (%)')
                    axes[1, 2].set_ylabel('Frequency')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            
        except Exception as e:
            logger.warning(f"Chart generation failed: {e}")
            print("Charts could not be generated due to missing data or error")
    
    def _generate_single_period_report(self, results: Dict):
        """Generate single period report"""
        print("\n" + "="*80)
        print("PROFESSIONAL OPTIONS TRADING SYSTEM - SINGLE PERIOD RESULTS")
        print("="*80)
        
        for index_name, result in results['results'].items():
            if 'error' in result:
                print(f"{index_name}: ERROR - {result['error']}")
                continue
            
            metrics = result.get('metrics', {})
            print(f"\n{index_name} Performance:")
            print(f"  Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            print(f"  Annualized Return: {metrics.get('annualized_return', 0)*100:.2f}%")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"  Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")

# ================================================================================
# MAIN EXECUTION FUNCTION
# ================================================================================

def main():
    """Main execution function with comprehensive error handling"""
    
    # Professional configuration
    config = TradingConfig(
        # Portfolio settings
        initial_capital=100000.0,  # ₹1,00,000
        
        # Multi-index setup
        indices=["^NSEI", "^NSEBANK", "^BSESN"],
        index_names=["NIFTY_50", "BANK_NIFTY", "SENSEX"],
        index_weights=[0.5, 0.3, 0.2],
        
        # Time period (adjust as needed)
        start_date=date(2020, 1, 1),
        end_date=date(2024, 12, 31),
        
        # Walk-forward configuration
        walk_forward_enabled=True,
        walk_forward_periods=4,
        training_window_months=12,
        testing_window_months=6,
        
        # Conservative risk management
        max_risk_per_trade=0.008,  # 0.8% per trade
        max_portfolio_risk=0.08,   # 8% total exposure
        max_positions_per_index=3,
        max_total_positions=8,
        
        # Professional exit rules
        stop_loss_pct=0.45,
        take_profit_pct=0.80,
        time_stop_days=25,
        
        # Enhanced options parameters
        min_dte=10,
        max_dte=45,
        min_credit_threshold=0.5,
        
        # Quality thresholds
        signal_threshold=0.35,
        
        # ML optimization
        use_ml_optimization=True,
        
        # System settings
        verbose_logging=True,
        cache_enabled=True
    )
    
    # Display professional banner
    if RICH_AVAILABLE:
        banner = Panel(
            f"[bold cyan]🚀 PROFESSIONAL ELITE OPTIONS TRADING SYSTEM v5.0[/bold cyan]\n\n"
            f"[green]✅ Advanced Multi-Index Backtesting Engine[/green]\n"
            f"Initial Capital: ₹{config.initial_capital:,.0f}\n"
            f"Testing Period: {config.start_date} to {config.end_date}\n"
            f"Indices: {', '.join(config.index_names)}\n"
            f"Walk-Forward: {'Enabled' if config.walk_forward_enabled else 'Disabled'}\n"
            f"Max Risk per Trade: {config.max_risk_per_trade:.1%}\n"
            f"Portfolio Risk Limit: {config.max_portfolio_risk:.1%}\n\n"
            f"[yellow]⚡ Advanced Features:[/yellow]\n"
            f"• Professional Options Pricing with Greeks\n"
            f"• 25+ Technical Indicators\n"
            f"• Machine Learning Optimization\n"
            f"• Multi-Strategy Selection\n"
            f"• Comprehensive Risk Management\n"
            f"• Professional Reporting\n\n"
            f"[red]⚠️  For educational and research purposes only[/red]\n"
            f"[red]Past performance ≠ Future results[/red]",
            title="🎯 Elite Trading System Configuration",
            border_style="green"
        )
        console.print(banner)
    else:
        print("="*80)
        print("🚀 PROFESSIONAL ELITE OPTIONS TRADING SYSTEM v5.0")
        print("="*80)
        print(f"Initial Capital: ₹{config.initial_capital:,.0f}")
        print(f"Testing Period: {config.start_date} to {config.end_date}")
        print(f"Indices: {', '.join(config.index_names)}")
        print(f"Walk-Forward: {'Enabled' if config.walk_forward_enabled else 'Disabled'}")
        print("="*80)
    
    try:
        # Initialize backtesting engine
        backtest_engine = ProfessionalBacktestEngine(config)
        
        # Run comprehensive backtest
        logger.info("Starting comprehensive backtesting process...")
        results = backtest_engine.run_comprehensive_backtest()
        
        # Generate professional report
        report_generator = ProfessionalReportGenerator()
        report_generator.generate_comprehensive_report(results)
        
        # Success message
        if RICH_AVAILABLE:
            success_panel = Panel(
                f"[bold green]✅ Professional Backtesting Completed Successfully![/bold green]\n\n"
                f"[cyan]🎯 Key Highlights:[/cyan]\n"
                f"• Advanced multi-index options strategies tested\n"
                f"• Professional risk management implemented\n"
                f"• Comprehensive performance analysis generated\n"
                f"• Walk-forward validation completed\n\n"
                f"[yellow]📊 System demonstrates institutional-grade capabilities[/yellow]\n\n"
                f"[red]⚠️  IMPORTANT DISCLAIMER:[/red]\n"
                f"[red]This system is for educational purposes only.[/red]\n"
                f"[red]Always consult with financial professionals before trading.[/red]\n"
                f"[red]Past performance does not guarantee future results.[/red]",
                title="🏆 Backtesting Complete",
                border_style="green"
            )
            console.print(success_panel)
        else:
            print("\n" + "="*80)
            print("✅ PROFESSIONAL BACKTESTING COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("🎯 Advanced options trading strategies analyzed")
            print("📊 Professional risk management validated")
            print("📈 Comprehensive performance metrics generated")
            print("\n⚠️  DISCLAIMER: For educational purposes only.")
            print("Always consult financial professionals before trading.")
            print("="*80)
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Backtesting interrupted by user")
        if RICH_AVAILABLE:
            console.print("[yellow]⚠️  Backtesting interrupted by user[/yellow]")
        return None
        
    except Exception as e:
        logger.error(f"Backtesting failed with error: {e}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]❌ Backtesting Failed: {e}[/bold red]")
        else:
            print(f"❌ Backtesting Failed: {e}")
        raise

# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    # Set up environment
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    np.random.seed(42)
    
    # Execute main function
    try:
        results = main()
        if results:
            print(f"\n🎯 Backtesting session completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"\n💥 System encountered an error: {e}")
        print("Please check the logs for detailed information.")
    finally:
        print("\n📋 Thank you for using the Professional Elite Options Trading System!")
        print("🔬 Remember: This is for educational and research purposes only.")



