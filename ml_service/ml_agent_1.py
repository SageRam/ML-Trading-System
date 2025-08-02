# ml_agent_enhanced.py - Production-Ready ML Analytics Service
import asyncio
import json
import logging
import os
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import redis.asyncio as redis
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import MetaTrader5 as mt5
import requests
from textblob import TextBlob
import yfinance as yf
import joblib
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configuration with production services
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:xPgernhZHDqzwMZZSusJrpPuUxZyUYoR@ballast.proxy.rlwy.net:56300/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://default:vanoEzlkdWoeXeSbOYIrtagILrJaugmb@yamabiko.proxy.rlwy.net:22322")
API_URL = os.getenv("API_URL", "https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app")
MODEL_UPDATE_INTERVAL = int(os.getenv("MODEL_UPDATE_INTERVAL", "3600"))  # 1 hour
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Trading signal types
class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class TradingSignal:
    symbol: str
    direction: SignalType
    entry_price: float
    stop_loss: float
    take_profit_levels: List[float]
    confidence: float
    pattern: str
    risk_reward_ratio: float
    market_context: Dict
    timestamp: datetime
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['direction'] = self.direction.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

class EnhancedTechnicalIndicators:
    """Advanced technical indicators calculator with additional features"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators with error handling"""
        try:
            # Validate input
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns. Expected: {required_cols}, Got: {df.columns.tolist()}")
                return df

            # Convert to numpy arrays for efficiency
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            volume = df['volume'].values.astype(np.float64)
            open_prices = df['open'].values.astype(np.float64)

            # Trend Indicators
            df['sma_10'] = talib.SMA(close, timeperiod=10)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_200'] = talib.SMA(close, timeperiod=200)
            
            df['ema_9'] = talib.EMA(close, timeperiod=9)
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            
            # TEMA (Triple Exponential Moving Average)
            df['tema'] = talib.TEMA(close, timeperiod=30)
            
            # KAMA (Kaufman Adaptive Moving Average)
            df['kama'] = talib.KAMA(close, timeperiod=30)

            # MACD variations
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close)
            df['macd_12_26'], df['macd_signal_9'], _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

            # Momentum Indicators
            df['rsi_14'] = talib.RSI(close, timeperiod=14)
            df['rsi_9'] = talib.RSI(close, timeperiod=9)
            df['rsi_25'] = talib.RSI(close, timeperiod=25)
            
            # Stochastic variations
            df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close)
            df['stoch_rsi_k'], df['stoch_rsi_d'] = talib.STOCHRSI(close)
            
            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close)
            
            # Ultimate Oscillator
            df['ultimate_osc'] = talib.ULTOSC(high, low, close)
            
            # CCI (Commodity Channel Index)
            df['cci'] = talib.CCI(high, low, close)
            
            # MFI (Money Flow Index)
            df['mfi'] = talib.MFI(high, low, close, volume)
            
            # CMO (Chande Momentum Oscillator)
            df['cmo'] = talib.CMO(close)

            # Volatility Indicators
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_percent'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # ATR variations
            df['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
            df['atr_20'] = talib.ATR(high, low, close, timeperiod=20)
            df['natr'] = talib.NATR(high, low, close)  # Normalized ATR
            
            # Keltner Channels
            kc_middle = df['ema_20'] = talib.EMA(close, timeperiod=20)
            kc_atr = df['atr_20']
            df['kc_upper'] = kc_middle + (2 * kc_atr)
            df['kc_lower'] = kc_middle - (2 * kc_atr)
            
            # Donchian Channels
            df['donchian_upper'] = df['high'].rolling(window=20).max()
            df['donchian_lower'] = df['low'].rolling(window=20).min()
            df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

            # Trend Strength Indicators
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['adxr'] = talib.ADXR(high, low, close)
            df['plus_di'] = talib.PLUS_DI(high, low, close)
            df['minus_di'] = talib.MINUS_DI(high, low, close)
            df['dx'] = talib.DX(high, low, close)
            
            # Aroon
            df['aroon_up'], df['aroon_down'] = talib.AROON(high, low)
            df['aroon_osc'] = talib.AROONOSC(high, low)

            # Volume Indicators
            df['obv'] = talib.OBV(close, volume)
            df['ad'] = talib.AD(high, low, close, volume)
            df['adosc'] = talib.ADOSC(high, low, close, volume)
            df['volume_sma'] = talib.SMA(volume, timeperiod=20)
            df['volume_ratio'] = volume / df['volume_sma']
            
            # VWAP (Volume Weighted Average Price)
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Chaikin Money Flow
            clv = ((close - low) - (high - close)) / (high - low)
            clv = np.nan_to_num(clv, 0)
            df['cmf'] = (clv * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()

            # Price Action Features
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['close'].diff()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Volatility measures
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            df['volatility_50'] = df['price_change'].rolling(window=50).std()
            df['volatility_ratio'] = df['volatility_20'] / df['volatility_50']
            
            # Price momentum
            df['momentum_10'] = df['close'] - df['close'].shift(10)
            df['momentum_20'] = df['close'] - df['close'].shift(20)
            df['roc'] = talib.ROC(close)  # Rate of Change
            
            # Support/Resistance Levels
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['r3'] = df['high'] + 2 * (df['pivot'] - df['low'])
            df['s1'] = 2 * df['pivot'] - df['high']
            df['s2'] = df['pivot'] - (df['high'] - df['low'])
            df['s3'] = df['low'] - 2 * (df['high'] - df['pivot'])
            
            # Fibonacci Retracements (simplified)
            rolling_high = df['high'].rolling(window=50).max()
            rolling_low = df['low'].rolling(window=50).min()
            diff = rolling_high - rolling_low
            df['fib_0.236'] = rolling_high - 0.236 * diff
            df['fib_0.382'] = rolling_high - 0.382 * diff
            df['fib_0.5'] = rolling_high - 0.5 * diff
            df['fib_0.618'] = rolling_high - 0.618 * diff
            
            # Market Structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            df['inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
            
            # Normalized indicators for better ML performance
            df['rsi_normalized'] = df['rsi_14'] / 100
            df['stoch_normalized'] = df['stoch_k'] / 100
            df['adx_normalized'] = df['adx'] / 100
            df['mfi_normalized'] = df['mfi'] / 100
            
            # Feature engineering for ML
            df['trend_score'] = (
                (df['close'] > df['sma_20']).astype(int) +
                (df['close'] > df['sma_50']).astype(int) +
                (df['sma_20'] > df['sma_50']).astype(int) +
                (df['macd'] > df['macd_signal']).astype(int)
            ) / 4
            
            # Market regime detection
            df['trend_strength'] = df['adx']
            df['is_trending'] = (df['adx'] > 25).astype(int)
            df['is_bullish'] = (df['close'] > df['sma_50']).astype(int)
            df['is_volatile'] = (df['volatility_20'] > df['volatility_20'].rolling(window=50).mean()).astype(int)
            
            # Clean up NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            logger.info(f"Calculated {len(df.columns)} technical indicators")
            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            logger.error(traceback.format_exc())
            return df

class AdvancedPatternRecognition:
    """Enhanced pattern recognition with machine learning integration"""

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, float]:
        """Detect trading patterns with confidence scores"""
        patterns = {}
        
        try:
            if len(df) < 100:
                logger.warning("Insufficient data for pattern detection")
                return patterns

            # Convert to numpy arrays
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            open_prices = df['open'].values

            # Candlestick patterns (using last 50 candles for scoring)
            window = min(50, len(df))
            
            # Bullish patterns
            patterns['hammer'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLHAMMER(open_prices, high, low, close), window
            )
            patterns['inverted_hammer'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLINVERTEDHAMMER(open_prices, high, low, close), window
            )
            patterns['doji'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLDOJI(open_prices, high, low, close), window
            )
            patterns['dragonfly_doji'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLDRAGONFLYDOJI(open_prices, high, low, close), window
            )
            patterns['engulfing_bull'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLENGULFING(open_prices, high, low, close), window
            )
            patterns['morning_star'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLMORNINGSTAR(open_prices, high, low, close), window
            )
            patterns['three_white_soldiers'] = AdvancedPatternRecognition._score_pattern(
                talib.CDL3WHITESOLDIERS(open_prices, high, low, close), window
            )
            patterns['bullish_harami'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLHARAMI(open_prices, high, low, close), window
            )
            
            # Bearish patterns
            patterns['hanging_man'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLHANGINGMAN(open_prices, high, low, close), window
            )
            patterns['shooting_star'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLSHOOTINGSTAR(open_prices, high, low, close), window
            )
            patterns['evening_star'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLEVENINGSTAR(open_prices, high, low, close), window
            )
            patterns['three_black_crows'] = AdvancedPatternRecognition._score_pattern(
                talib.CDL3BLACKCROWS(open_prices, high, low, close), window
            )
            patterns['bearish_harami'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLHARAMI(open_prices, high, low, close), window
            )
            patterns['dark_cloud_cover'] = AdvancedPatternRecognition._score_pattern(
                talib.CDLDARKCLOUDCOVER(open_prices, high, low, close), window
            )

            # Chart patterns
            patterns.update(AdvancedPatternRecognition._detect_chart_patterns(df))
            
            # Harmonic patterns
            patterns.update(AdvancedPatternRecognition._detect_harmonic_patterns(df))
            
            # Custom patterns
            patterns.update(AdvancedPatternRecognition._detect_custom_patterns(df))

            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return patterns

    @staticmethod
    def _score_pattern(pattern_array: np.ndarray, window: int) -> float:
        """Calculate pattern confidence score"""
        if len(pattern_array) < window:
            return 0.0
        
        # Count occurrences in the window
        occurrences = np.sum(np.abs(pattern_array[-window:]) > 0)
        
        # Weight recent occurrences more heavily
        weighted_score = 0.0
        weights = np.exp(np.linspace(-1, 0, window))
        weights = weights / weights.sum()
        
        for i in range(window):
            if abs(pattern_array[-(window-i)]) > 0:
                weighted_score += weights[i] * np.sign(pattern_array[-(window-i)])
        
        return float(np.clip(weighted_score, -1, 1))

    @staticmethod
    def _detect_chart_patterns(df: pd.DataFrame) -> Dict[str, float]:
        """Detect chart patterns like double top/bottom, triangles, etc."""
        patterns = {}
        
        try:
            # Double Top/Bottom
            patterns['double_top'] = AdvancedPatternRecognition._detect_double_top_bottom(df, 'top')
            patterns['double_bottom'] = AdvancedPatternRecognition._detect_double_top_bottom(df, 'bottom')
            
            # Head and Shoulders
            patterns['head_shoulders'] = AdvancedPatternRecognition._detect_head_shoulders(df)
            patterns['inverse_head_shoulders'] = AdvancedPatternRecognition._detect_head_shoulders(df, inverse=True)
            
            # Triangles
            patterns['ascending_triangle'] = AdvancedPatternRecognition._detect_triangle(df, 'ascending')
            patterns['descending_triangle'] = AdvancedPatternRecognition._detect_triangle(df, 'descending')
            patterns['symmetric_triangle'] = AdvancedPatternRecognition._detect_triangle(df, 'symmetric')
            
            # Wedges
            patterns['rising_wedge'] = AdvancedPatternRecognition._detect_wedge(df, 'rising')
            patterns['falling_wedge'] = AdvancedPatternRecognition._detect_wedge(df, 'falling')
            
            # Channels
            patterns['channel_up'] = AdvancedPatternRecognition._detect_channel(df, 'up')
            patterns['channel_down'] = AdvancedPatternRecognition._detect_channel(df, 'down')
            
            # Flags and Pennants
            patterns['bull_flag'] = AdvancedPatternRecognition._detect_flag(df, 'bull')
            patterns['bear_flag'] = AdvancedPatternRecognition._detect_flag(df, 'bear')
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {e}")
            return patterns

    @staticmethod
    def _detect_double_top_bottom(df: pd.DataFrame, pattern_type: str) -> float:
        """Enhanced double top/bottom detection"""
        try:
            if len(df) < 100:
                return 0.0

            price_col = 'high' if pattern_type == 'top' else 'low'
            prices = df[price_col].values
            
            # Find local extrema
            window = 10
            if pattern_type == 'top':
                extrema = talib.MAX(prices, timeperiod=window)
                mask = prices == extrema
            else:
                extrema = talib.MIN(prices, timeperiod=window)
                mask = prices == extrema
            
            # Get indices of extrema
            extrema_indices = np.where(mask)[0]
            
            if len(extrema_indices) < 2:
                return 0.0
            
            # Check last two extrema
            last_two = extrema_indices[-2:]
            height_diff = abs(prices[last_two[0]] - prices[last_two[1]])
            time_diff = last_two[1] - last_two[0]
            
            # Calculate pattern score
            atr = df['atr_14'].iloc[-1] if 'atr_14' in df.columns else df['close'].pct_change().std() * df['close'].iloc[-1]
            
            if height_diff < atr * 2 and 20 < time_diff < 100:
                score = 1.0 - (height_diff / (atr * 2))
                
                # Verify neckline
                neckline_prices = prices[last_two[0]:last_two[1]]
                if pattern_type == 'top':
                    neckline = neckline_prices.min()
                    if prices[-1] < neckline:
                        score *= 1.2  # Pattern confirmed
                else:
                    neckline = neckline_prices.max()
                    if prices[-1] > neckline:
                        score *= 1.2  # Pattern confirmed
                
                return float(np.clip(score, 0, 1))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error detecting double {pattern_type}: {e}")
            return 0.0

    @staticmethod
    def _detect_head_shoulders(df: pd.DataFrame, inverse: bool = False) -> float:
        """Detect head and shoulders pattern"""
        try:
            if len(df) < 150:
                return 0.0
            
            prices = df['low' if inverse else 'high'].values
            
            # Find peaks/troughs
            window = 15
            if inverse:
                extrema = talib.MIN(prices, timeperiod=window)
                peaks = np.where(prices == extrema)[0]
            else:
                extrema = talib.MAX(prices, timeperiod=window)
                peaks = np.where(prices == extrema)[0]
            
            if len(peaks) < 5:
                return 0.0
            
            # Check last 5 peaks for H&S pattern
            last_peaks = peaks[-5:]
            peak_values = prices[last_peaks]
            
            # Classic H&S: left shoulder < head > right shoulder
            # Shoulders should be roughly equal
            left_shoulder = peak_values[0]
            left_trough = peak_values[1]
            head = peak_values[2]
            right_trough = peak_values[3]
            right_shoulder = peak_values[4]
            
            if inverse:
                condition = (head < left_shoulder and head < right_shoulder and
                           abs(left_shoulder - right_shoulder) < abs(head - left_shoulder) * 0.3)
            else:
                condition = (head > left_shoulder and head > right_shoulder and
                           abs(left_shoulder - right_shoulder) < abs(head - left_shoulder) * 0.3)
            
            if condition:
                # Calculate score based on symmetry and neckline
                symmetry = 1.0 - abs(left_shoulder - right_shoulder) / max(abs(head - left_shoulder), 0.0001)
                neckline_consistency = 1.0 - abs(left_trough - right_trough) / max(abs(head - left_trough), 0.0001)
                
                score = (symmetry + neckline_consistency) / 2
                return float(np.clip(score, 0, 1))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error detecting head and shoulders: {e}")
            return 0.0

    @staticmethod
    def _detect_triangle(df: pd.DataFrame, triangle_type: str) -> float:
        """Detect triangle patterns"""
        try:
            if len(df) < 50:
                return 0.0
            
            highs = df['high'].values[-50:]
            lows = df['low'].values[-50:]
            
            # Fit trend lines
            x = np.arange(len(highs))
            high_slope, high_intercept = np.polyfit(x, highs, 1)
            low_slope, low_intercept = np.polyfit(x, lows, 1)
            
            # Calculate R-squared for trend lines
            high_r2 = np.corrcoef(x, highs)[0, 1] ** 2
            low_r2 = np.corrcoef(x, lows)[0, 1] ** 2
            
            min_r2 = 0.7  # Minimum R-squared for valid trend line
            
            if high_r2 < min_r2 or low_r2 < min_r2:
                return 0.0
            
            # Check triangle type
            if triangle_type == 'ascending':
                # Flat top, rising bottom
                if abs(high_slope) < 0.001 and low_slope > 0:
                    return float((high_r2 + low_r2) / 2)
            elif triangle_type == 'descending':
                # Falling top, flat bottom
                if high_slope < 0 and abs(low_slope) < 0.001:
                    return float((high_r2 + low_r2) / 2)
            elif triangle_type == 'symmetric':
                # Converging lines
                if high_slope < 0 and low_slope > 0 and abs(high_slope + low_slope) < 0.001:
                    return float((high_r2 + low_r2) / 2)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error detecting {triangle_type} triangle: {e}")
            return 0.0

    @staticmethod
    def _detect_wedge(df: pd.DataFrame, wedge_type: str) -> float:
        """Detect wedge patterns"""
        # Similar to triangle but both lines slope in same direction
        # Implementation would be similar to triangle detection
        return 0.0

    @staticmethod
    def _detect_channel(df: pd.DataFrame, direction: str) -> float:
        """Detect channel patterns"""
        # Parallel trend lines
        return 0.0

    @staticmethod
    def _detect_flag(df: pd.DataFrame, flag_type: str) -> float:
        """Detect flag patterns"""
        # Sharp move followed by consolidation
        return 0.0

    @staticmethod
    def _detect_harmonic_patterns(df: pd.DataFrame) -> Dict[str, float]:
        """Detect harmonic patterns (Gartley, Butterfly, etc.)"""
        patterns = {}
        
        # Simplified harmonic pattern detection
        # In practice, this would involve Fibonacci ratios and swing points
        patterns['gartley'] = 0.0
        patterns['butterfly'] = 0.0
        patterns['bat'] = 0.0
        patterns['crab'] = 0.0
        
        return patterns

    @staticmethod
    def _detect_custom_patterns(df: pd.DataFrame) -> Dict[str, float]:
        """Detect custom patterns specific to trading strategy"""
        patterns = {}
        
        try:
            # Trend continuation patterns
            if 'sma_20' in df.columns and 'sma_50' in df.columns:
                # Golden cross
                sma_20 = df['sma_20'].values
                sma_50 = df['sma_50'].values
                if len(sma_20) > 2:
                    if sma_20[-1] > sma_50[-1] and sma_20[-2] <= sma_50[-2]:
                        patterns['golden_cross'] = 1.0
                    elif sma_20[-1] < sma_50[-1] and sma_20[-2] >= sma_50[-2]:
                        patterns['death_cross'] = -1.0
            
            # Volume patterns
            if 'volume' in df.columns:
                vol = df['volume'].values
                vol_sma = talib.SMA(vol, timeperiod=20)
                if len(vol) > 0 and len(vol_sma) > 0:
                    # Volume spike
                    if vol[-1] > vol_sma[-1] * 2:
                        patterns['volume_spike'] = 1.0
                    # Volume dry up
                    elif vol[-1] < vol_sma[-1] * 0.5:
                        patterns['volume_dryup'] = -0.5
            
            # Momentum patterns
            if 'rsi_14' in df.columns:
                rsi = df['rsi_14'].values
                if len(rsi) > 5:
                    # RSI divergence
                    price_trend = np.polyfit(range(5), df['close'].values[-5:], 1)[0]
                    rsi_trend = np.polyfit(range(5), rsi[-5:], 1)[0]
                    
                    if price_trend > 0 and rsi_trend < 0:
                        patterns['bearish_divergence'] = -0.8
                    elif price_trend < 0 and rsi_trend > 0:
                        patterns['bullish_divergence'] = 0.8
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting custom patterns: {e}")
            return patterns

class EnhancedSentimentAnalyzer:
    """Advanced sentiment analysis with multiple data sources"""
    
    def __init__(self):
        self.news_cache = {}
        self.sentiment_history = []
        self.api_keys = {
            'newsapi': os.getenv('NEWS_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY'),
            'finnhub': os.getenv('FINNHUB_KEY')
        }

    async def analyze_news_sentiment(self, symbol: str = "USDJPY") -> Dict:
        """Comprehensive sentiment analysis from multiple sources"""
        try:
            # Check cache first
            cache_key = f"sentiment_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
            if cache_key in self.news_cache:
                return self.news_cache[cache_key]

            sentiments = []
            sources_data = {}

            # Fetch from multiple sources concurrently
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = []
                
                if self.api_keys['newsapi']:
                    futures.append(executor.submit(self._fetch_newsapi_sentiment, symbol))
                if self.api_keys['alpha_vantage']:
                    futures.append(executor.submit(self._fetch_alphavantage_sentiment, symbol))
                if self.api_keys['finnhub']:
                    futures.append(executor.submit(self._fetch_finnhub_sentiment, symbol))
                
                # If no API keys, use mock data
                if not futures:
                    futures.append(executor.submit(self._generate_mock_sentiment, symbol))
                
                for future in futures:
                    try:
                        source_name, sentiment_data = future.result(timeout=10)
                        if sentiment_data:
                            sentiments.append(sentiment_data['score'])
                            sources_data[source_name] = sentiment_data
                    except Exception as e:
                        logger.warning(f"Failed to fetch sentiment from source: {e}")

            # Calculate aggregate sentiment
            if sentiments:
                overall_sentiment = np.mean(sentiments)
                confidence = 1.0 - np.std(sentiments) if len(sentiments) > 1 else 0.7
            else:
                overall_sentiment = 0.0
                confidence = 0.0

            # Analyze themes and keywords
            themes = self._extract_themes(sources_data)
            
            sentiment_result = {
                'overall': float(overall_sentiment),
                'confidence': float(confidence),
                'sources': sources_data,
                'themes': themes,
                'sentiment_distribution': {
                    'positive': sum(1 for s in sentiments if s > 0.1) / max(len(sentiments), 1),
                    'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1) / max(len(sentiments), 1),
                    'negative': sum(1 for s in sentiments if s < -0.1) / max(len(sentiments), 1)
                },
                'last_updated': datetime.now().isoformat()
            }

            # Cache result
            self.news_cache[cache_key] = sentiment_result
            
            # Store in history for trend analysis
            self.sentiment_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'sentiment': overall_sentiment
            })
            
            # Keep only last 100 entries
            if len(self.sentiment_history) > 100:
                self.sentiment_history = self.sentiment_history[-100:]

            return sentiment_result

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'overall': 0.0,
                'confidence': 0.0,
                'sources': {},
                'themes': [],
                'sentiment_distribution': {'positive': 0, 'neutral': 1, 'negative': 0},
                'last_updated': datetime.now().isoformat()
            }

    def _fetch_newsapi_sentiment(self, symbol: str) -> Tuple[str, Dict]:
        """Fetch sentiment from NewsAPI"""
        try:
            # Implementation would call NewsAPI
            # For now, return mock data
            return self._generate_mock_sentiment(symbol)
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return "newsapi", None

    def _fetch_alphavantage_sentiment(self, symbol: str) -> Tuple[str, Dict]:
        """Fetch sentiment from Alpha Vantage"""
        try:
            # Implementation would call Alpha Vantage API
            return self._generate_mock_sentiment(symbol)
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
            return "alpha_vantage", None

    def _fetch_finnhub_sentiment(self, symbol: str) -> Tuple[str, Dict]:
        """Fetch sentiment from Finnhub"""
        try:
            # Implementation would call Finnhub API
            return self._generate_mock_sentiment(symbol)
        except Exception as e:
            logger.error(f"Finnhub error: {e}")
            return "finnhub", None

    def _generate_mock_sentiment(self, symbol: str) -> Tuple[str, Dict]:
        """Generate mock sentiment data for testing"""
        base_sentiment = np.random.uniform(-0.3, 0.3)
        
        # Add some market-hour bias
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 16:  # Market hours
            base_sentiment += np.random.uniform(-0.1, 0.1)
        
        sentiment_data = {
            'score': base_sentiment,
            'articles_analyzed': np.random.randint(5, 20),
            'keywords': ['forex', symbol.lower(), 'trading', 'market'],
            'summary': f"Mock sentiment analysis for {symbol}"
        }
        
        return "mock_source", sentiment_data

    def _extract_themes(self, sources_data: Dict) -> List[str]:
        """Extract common themes from sentiment data"""
        all_keywords = []
        for source, data in sources_data.items():
            if data and 'keywords' in data:
                all_keywords.extend(data['keywords'])
        
        # Count frequency and return top themes
        from collections import Counter
        theme_counts = Counter(all_keywords)
        return [theme for theme, _ in theme_counts.most_common(5)]

    def get_sentiment_trend(self, symbol: str, hours: int = 24) -> Dict:
        """Get sentiment trend over time"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        relevant_history = [
            entry for entry in self.sentiment_history
            if entry['symbol'] == symbol and entry['timestamp'] > cutoff_time
        ]
        
        if not relevant_history:
            return {'trend': 'neutral', 'change': 0.0}
        
        sentiments = [entry['sentiment'] for entry in relevant_history]
        
        # Calculate trend
        if len(sentiments) >= 2:
            x = np.arange(len(sentiments))
            slope, _ = np.polyfit(x, sentiments, 1)
            
            if slope > 0.001:
                trend = 'improving'
            elif slope < -0.001:
                trend = 'deteriorating'
            else:
                trend = 'stable'
            
            change = sentiments[-1] - sentiments[0] if sentiments else 0.0
        else:
            trend = 'insufficient_data'
            change = 0.0
        
        return {
            'trend': trend,
            'change': float(change),
            'current': float(sentiments[-1]) if sentiments else 0.0,
            'average': float(np.mean(sentiments)) if sentiments else 0.0,
            'volatility': float(np.std(sentiments)) if len(sentiments) > 1 else 0.0
        }

class EnhancedMLPredictor:
    """Advanced ML prediction engine with ensemble methods"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selector = None
        self.feature_columns = []
        self.selected_features = []
        self.is_trained = False
        self.model_dir = "models"
        self.performance_history = []
        
        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Enhanced feature preparation with engineering"""
        try:
            # Calculate all technical indicators
            df = EnhancedTechnicalIndicators.calculate_indicators(df)
            
            # Add pattern recognition features
            patterns = AdvancedPatternRecognition.detect_patterns(df)
            for pattern_name, confidence in patterns.items():
                df[f'pattern_{pattern_name}'] = confidence

            # Time-based features
            if df.index.dtype == 'int64':
                df.index = pd.to_datetime(df.index, unit='s')
            
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Market session features
            df['tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
            df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
            df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
            df['session_overlap'] = ((df['london_session'] == 1) & (df['ny_session'] == 1)).astype(int)

            # Lag features with multiple timeframes
            for lag in [1, 2, 3, 5, 10, 20, 50]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag) if 'rsi_14' in df.columns else 0
                df[f'momentum_lag_{lag}'] = df['momentum_10'].shift(lag) if 'momentum_10' in df.columns else 0

            # Rolling statistics with multiple windows
            for window in [5, 10, 20, 50]:
                # Price statistics
                df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
                df[f'close_std_{window}'] = df['close'].rolling(window).std()
                df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
                df[f'close_kurt_{window}'] = df['close'].rolling(window).kurt()
                
                # Volume statistics
                df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
                
                # High-low spread
                df[f'hl_spread_mean_{window}'] = (df['high'] - df['low']).rolling(window).mean()
                
                # Return statistics
                returns = df['close'].pct_change()
                df[f'return_mean_{window}'] = returns.rolling(window).mean()
                df[f'return_std_{window}'] = returns.rolling(window).std()
                df[f'return_skew_{window}'] = returns.rolling(window).skew()

            # Microstructure features
            df['bid_ask_spread'] = df['high'] - df['low']  # Proxy for spread
            df['price_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
            df['volume_price_trend'] = (df['volume'] * df['close'].pct_change()).cumsum()
            
            # Relative position indicators
            df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
            df['close_to_sma20'] = (df['close'] - df['sma_20']) / df['sma_20'] if 'sma_20' in df.columns else 0
            df['close_to_bb_upper'] = (df['bb_upper'] - df['close']) / df['close'] if 'bb_upper' in df.columns else 0
            df['close_to_bb_lower'] = (df['close'] - df['bb_lower']) / df['close'] if 'bb_lower' in df.columns else 0

            # Market regime features
            df['trend_consistency'] = (
                (df['close'] > df['sma_10']).astype(int) +
                (df['close'] > df['sma_20']).astype(int) +
                (df['close'] > df['sma_50']).astype(int)
            ) / 3 if all(col in df.columns for col in ['sma_10', 'sma_20', 'sma_50']) else 0.5

            # Interaction features
            if 'rsi_14' in df.columns and 'macd' in df.columns:
                df['rsi_macd_interaction'] = df['rsi_14'] * np.sign(df['macd'])
            
            if 'adx' in df.columns and 'atr_14' in df.columns:
                df['trend_volatility_ratio'] = df['adx'] / (df['atr_14'] + 1e-10)

            # Target variable (for training)
            if is_training:
                # Multiple prediction horizons
                for horizon in [1, 5, 10]:
                    df[f'future_return_{horizon}'] = df['close'].shift(-horizon) / df['close'] - 1
                
                # Primary target: direction of next candle
                df['target'] = (df['future_return_1'] > 0.0001).astype(int)  # 1 pip threshold
                
                # Multi-class target for more nuanced predictions
                df['target_multiclass'] = pd.cut(
                    df['future_return_1'],
                    bins=[-np.inf, -0.002, -0.0001, 0.0001, 0.002, np.inf],
                    labels=['strong_sell', 'sell', 'neutral', 'buy', 'strong_buy']
                )

            # Clean up
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)

            return df

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            logger.error(traceback.format_exc())
            return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> List[str]:
        """Select most important features using statistical tests"""
        try:
            selector = SelectKBest(score_func=f_classif, k=min(n_features, len(X.columns)))
            selector.fit(X, y)
            
            # Get selected feature names
            feature_mask = selector.get_support()
            selected = X.columns[feature_mask].tolist()
            
            # Also include some must-have features
            must_have = ['close', 'volume', 'rsi_14', 'macd', 'atr_14', 'adx']
            for feature in must_have:
                if feature in X.columns and feature not in selected:
                    selected.append(feature)
            
            return selected[:n_features]
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X.columns.tolist()[:n_features]

    def train_models(self, df: pd.DataFrame) -> Dict:
        """Train ensemble of models with cross-validation"""
        try:
            logger.info("Starting model training...")
            
            # Prepare features
            df = self.prepare_features(df, is_training=True)
            
            # Remove unnecessary columns
            feature_cols = [col for col in df.columns if col not in [
                'open', 'high', 'low', 'close', 'volume', 'time',
                'future_return_1', 'future_return_5', 'future_return_10',
                'target', 'target_multiclass'
            ] and not col.startswith('future_')]
            
            # Prepare data
            X = df[feature_cols].fillna(0)
            y = df['target'].fillna(0)
            
            # Remove invalid samples
            mask = ~(y.isna() | np.isinf(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 500:
                logger.warning("Insufficient data for training")
                return {"status": "error", "message": "Insufficient data"}
            
            # Feature selection
            self.selected_features = self.select_features(X, y, n_features=50)
            X_selected = X[self.selected_features]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features - try both scalers
            scalers = {
                'standard': StandardScaler(),
                'robust': RobustScaler()
            }
            
            best_score = 0
            best_config = None
            
            for scaler_name, scaler in scalers.items():
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train multiple models
                models = {
                    'rf': RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        min_samples_split=20,
                        min_samples_leaf=10,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'gb': GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=5,
                        min_samples_split=20,
                        random_state=42
                    ),
                    'rf_balanced': RandomForestClassifier(
                        n_estimators=200,
                        max_depth=15,
                        class_weight='balanced',
                        random_state=42,
                        n_jobs=-1
                    )
                }
                
                trained_models = {}
                scores = {}
                
                for name, model in models.items():
                    # Train with cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    logger.info(f"{name} CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                    
                    # Train on full training set
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test_scaled)
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                    
                    trained_models[name] = model
                    scores[name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'cv_score': cv_scores.mean()
                    }
                    
                    logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                
                # Create ensemble
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft'
                )
                ensemble.fit(X_train_scaled, y_train)
                
                ensemble_score = accuracy_score(y_test, ensemble.predict(X_test_scaled))
                scores['ensemble'] = {'accuracy': ensemble_score}
                
                if ensemble_score > best_score:
                    best_score = ensemble_score
                    best_config = {
                        'scaler': scaler,
                        'scaler_name': scaler_name,
                        'models': trained_models,
                        'ensemble': ensemble,
                        'scores': scores
                    }
            
            # Save best configuration
            self.scalers['main'] = best_config['scaler']
            self.models = best_config['models']
            self.models['ensemble'] = best_config['ensemble']
            self.feature_columns = self.selected_features
            self.is_trained = True
            
            # Save models
            self._save_models()
            
            # Store performance metrics
            self.performance_history.append({
                'timestamp': datetime.now(),
                'scores': best_config['scores'],
                'n_samples': len(X_train),
                'n_features': len(self.selected_features)
            })
            
            return {
                "status": "success",
                "best_model": "ensemble",
                "accuracy": best_score,
                "models_trained": list(self.models.keys()),
                "features_selected": len(self.selected_features),
                "training_samples": len(X_train),
                "scores": best_config['scores']
            }

        except Exception as e:
            logger.error(f"Error training models: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def predict(self, df: pd.DataFrame, use_ensemble: bool = True) -> Dict:
        """Make predictions with confidence intervals"""
        try:
            if not self.is_trained:
                return {"status": "error", "message": "Models not trained"}

            # Prepare features
            df = self.prepare_features(df, is_training=False)
            
            # Get latest data point
            latest_data = df[self.selected_features].iloc[-1:].fillna(0)
            
            # Scale features
            latest_scaled = self.scalers['main'].transform(latest_data)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for name, model in self.models.items():
                if name != 'ensemble' or use_ensemble:
                    pred = model.predict(latest_scaled)[0]
                    prob = model.predict_proba(latest_scaled)[0]
                    
                    predictions[name] = int(pred)
                    probabilities[name] = {
                        'sell': float(prob[0]),
                        'buy': float(prob[1])
                    }
            
            # Aggregate predictions
            if use_ensemble and 'ensemble' in predictions:
                final_prediction = predictions['ensemble']
                final_probability = probabilities['ensemble']
            else:
                # Majority vote
                votes = list(predictions.values())
                final_prediction = 1 if sum(votes) > len(votes) / 2 else 0
                
                # Average probabilities
                buy_probs = [p['buy'] for p in probabilities.values()]
                sell_probs = [p['sell'] for p in probabilities.values()]
                final_probability = {
                    'buy': float(np.mean(buy_probs)),
                    'sell': float(np.mean(sell_probs))
                }
            
            # Calculate confidence
            confidence = float(max(final_probability.values()))
            
            # Get feature importances (from RF model)
            feature_importance = {}
            if 'rf' in self.models:
                importances = self.models['rf'].feature_importances_
                for feature, importance in zip(self.selected_features, importances):
                    if importance > 0.01:  # Only significant features
                        feature_importance[feature] = float(importance)
            
            # Determine signal strength
            prob_diff = abs(final_probability['buy'] - final_probability['sell'])
            if prob_diff > 0.3:
                strength = 'strong'
            elif prob_diff > 0.15:
                strength = 'moderate'
            else:
                strength = 'weak'

            return {
                "status": "success",
                "prediction": int(final_prediction),
                "signal": "BUY" if final_prediction == 1 else "SELL",
                "probability": final_probability,
                "confidence": confidence,
                "strength": strength,
                "model_predictions": predictions,
                "model_probabilities": probabilities,
                "top_features": dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:10])
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _save_models(self):
        """Save trained models and configuration"""
        try:
            # Save main configuration
            config = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'selected_features': self.selected_features,
                'performance_history': self.performance_history
            }
            
            model_path = os.path.join(self.model_dir, 'ml_models.pkl')
            joblib.dump(config, model_path)
            
            logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def load_models(self) -> bool:
        """Load trained models"""
        try:
            model_path = os.path.join(self.model_dir, 'ml_models.pkl')
            
            if os.path.exists(model_path):
                config = joblib.load(model_path)
                
                self.models = config['models']
                self.scalers = config['scalers']
                self.feature_columns = config.get('feature_columns', [])
                self.selected_features = config.get('selected_features', self.feature_columns)
                self.performance_history = config.get('performance_history', [])
                self.is_trained = True
                
                logger.info("Models loaded successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

class TradingMLService:
    """Enhanced ML service with advanced features"""

    def __init__(self):
        self.redis_client = None
        self.db_engine = None
        self.predictor = EnhancedMLPredictor()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.is_initialized = False
        
        # Performance tracking
        self.signal_history = []
        self.prediction_accuracy = []

    async def initialize(self):
        """Initialize all components"""
        try:
            # Database connection with Railway PostgreSQL
            self.db_engine = create_engine(
                DATABASE_URL,
                poolclass=NullPool,  # Disable pooling for async compatibility
                echo=False,
                connect_args={
                    "sslmode": "require",  # Railway requires SSL
                    "connect_timeout": 30
                }
            )
            
            # Redis connection with Railway Redis
            self.redis_client = await redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=30,
                socket_timeout=30
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Load ML models
            if not self.predictor.load_models():
                logger.info("No existing models found, will train on first run")
            
            self.is_initialized = True
            logger.info("Trading ML Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing service: {e}")
            self.is_initialized = False

    async def get_market_data(self, symbol: str = "USDJPY", timeframe: str = "M15", count: int = 1000) -> pd.DataFrame:
        """Get market data with fallback options"""
        try:
            # Try primary API with Vercel endpoint
            try:
                # Add timeout and headers for Vercel
                headers = {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.get(
                        f"{API_URL}/api/market-data/{symbol}",
                        params={"timeframe": timeframe, "count": count},
                        timeout=30,  # Increased timeout for Vercel
                        headers=headers
                    )
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data:
                        df = pd.DataFrame(data['data'])
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                        return df
            except Exception as e:
                logger.warning(f"Failed to fetch from Vercel API: {e}")
            
            # Fallback to Yahoo Finance
            ticker_map = {
                "USDJPY": "JPY=X",
                "EURUSD": "EURUSD=X",
                "GBPUSD": "GBPUSD=X",
                "AUDUSD": "AUDUSD=X",
                "USDCAD": "CAD=X",
                "NZDUSD": "NZDUSD=X"
            }
            
            ticker = ticker_map.get(symbol, "JPY=X")
            
            # Determine period based on timeframe and count
            if timeframe == "M15":
                period = "1mo"  # Approximate
            elif timeframe == "H1":
                period = "3mo"
            elif timeframe == "D1":
                period = "1y"
            else:
                period = "1mo"
            
            yf_data = yf.download(
                ticker,
                period=period,
                interval="15m" if timeframe == "M15" else "1h",
                progress=False
            )
            
            if not yf_data.empty:
                yf_data.columns = [col.lower() for col in yf_data.columns]
                yf_data['volume'] = yf_data.get('volume', 1000)
                
                # Ensure we have all required columns
                required = ['open', 'high', 'low', 'close', 'volume']
                for col in required:
                    if col not in yf_data.columns:
                        yf_data[col] = yf_data['close']  # Use close as fallback
                
                return yf_data.tail(count)
            
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return pd.DataFrame()

    async def analyze_market(self, symbol: str = "USDJPY", save_to_cache: bool = True) -> Dict:
        """Comprehensive market analysis with all components"""
        try:
            analysis_start = datetime.now()
            
            # Get market data
            df = await self.get_market_data(symbol)
            if df.empty:
                logger.error(f"No market data available for {symbol}")
                return {"status": "error", "message": "No market data available"}
            
            # Technical analysis
            df_with_indicators = EnhancedTechnicalIndicators.calculate_indicators(df)
            
            # Pattern recognition
            patterns = AdvancedPatternRecognition.detect_patterns(df_with_indicators)
            
            # ML prediction
            ml_prediction = self.predictor.predict(df_with_indicators) if self.predictor.is_trained else None
            
            # Sentiment analysis
            sentiment = await self.sentiment_analyzer.analyze_news_sentiment(symbol)
            sentiment_trend = self.sentiment_analyzer.get_sentiment_trend(symbol)
            
            # Market regime analysis
            market_regime = self._analyze_market_regime(df_with_indicators)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(df_with_indicators)
            
            # Generate trading signal
            signal = await self._generate_enhanced_signal(
                df_with_indicators, patterns, ml_prediction, sentiment, market_regime
            )
            
            # Get current market state
            latest = df_with_indicators.iloc[-1]
            
            # Compile comprehensive analysis
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "analysis_time_ms": (datetime.now() - analysis_start).total_seconds() * 1000,
                
                "price": {
                    "current": float(latest['close']),
                    "open": float(latest['open']),
                    "high": float(latest['high']),
                    "low": float(latest['low']),
                    "change": float(latest['close'] - df_with_indicators['close'].iloc[-2]),
                    "change_pct": float((latest['close'] - df_with_indicators['close'].iloc[-2]) / 
                                      df_with_indicators['close'].iloc[-2] * 100),
                    "volume": float(latest['volume'])
                },
                
                "technical_indicators": {
                    # Trend
                    "sma_20": float(latest.get('sma_20', 0)),
                    "sma_50": float(latest.get('sma_50', 0)),
                    "ema_12": float(latest.get('ema_12', 0)),
                    "ema_26": float(latest.get('ema_26', 0)),
                    
                    # Momentum
                    "rsi_14": float(latest.get('rsi_14', 50)),
                    "rsi_9": float(latest.get('rsi_9', 50)),
                    "macd": float(latest.get('macd', 0)),
                    "macd_signal": float(latest.get('macd_signal', 0)),
                    "macd_histogram": float(latest.get('macd_hist', 0)),
                    "stoch_k": float(latest.get('stoch_k', 50)),
                    "stoch_d": float(latest.get('stoch_d', 50)),
                    
                    # Volatility
                    "atr_14": float(latest.get('atr_14', 0)),
                    "bb_upper": float(latest.get('bb_upper', 0)),
                    "bb_middle": float(latest.get('bb_middle', 0)),
                    "bb_lower": float(latest.get('bb_lower', 0)),
                    "bb_width": float(latest.get('bb_width', 0)),
                    
                    # Trend Strength
                    "adx": float(latest.get('adx', 0)),
                    "plus_di": float(latest.get('plus_di', 0)),
                    "minus_di": float(latest.get('minus_di', 0)),
                    
                    # Volume
                    "obv": float(latest.get('obv', 0)),
                    "volume_ratio": float(latest.get('volume_ratio', 1))
                },
                
                "patterns": {
                    name: float(confidence) 
                    for name, confidence in patterns.items() 
                    if abs(confidence) > 0.1
                },
                
                "ml_prediction": ml_prediction if ml_prediction else {
                    "status": "not_available",
                    "message": "Model not trained yet"
                },
                
                "sentiment": {
                    "current": sentiment,
                    "trend": sentiment_trend
                },
                
                "market_regime": market_regime,
                "risk_metrics": risk_metrics,
                "signal": signal.to_dict() if signal else None,
                
                "market_conditions": {
                    "volatility": float(df_with_indicators['volatility_20'].iloc[-1]) 
                                 if 'volatility_20' in df_with_indicators.columns else 0.0,
                    "trend_strength": float(latest.get('trend_strength', 0)),
                    "liquidity": "high" if latest['volume'] > df_with_indicators['volume'].mean() else "normal"
                }
            }
            
            # Cache analysis
            if save_to_cache and self.redis_client:
                try:
                    await self.redis_client.setex(
                        f"analysis:{symbol}:{timeframe}",
                        300,  # 5 minutes TTL
                        json.dumps(analysis, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache analysis: {e}")
            
            return analysis

        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def _analyze_market_regime(self, df: pd.DataFrame) -> Dict:
        """Analyze current market regime"""
        try:
            if len(df) < 50:
                return {"type": "unknown", "confidence": 0.0}
            
            latest = df.iloc[-1]
            
            # Trend analysis
            sma_20 = latest.get('sma_20', latest['close'])
            sma_50 = latest.get('sma_50', latest['close'])
            adx = latest.get('adx', 0)
            
            # Volatility analysis
            current_volatility = df['volatility_20'].iloc[-1] if 'volatility_20' in df.columns else 0
            avg_volatility = df['volatility_20'].mean() if 'volatility_20' in df.columns else 0
            
            # Determine regime
            regime_scores = {
                'trending_up': 0,
                'trending_down': 0,
                'ranging': 0,
                'volatile': 0,
                'quiet': 0
            }
            
            # Trend indicators
            if adx > 25:
                if latest['close'] > sma_20 > sma_50:
                    regime_scores['trending_up'] += 0.6
                elif latest['close'] < sma_20 < sma_50:
                    regime_scores['trending_down'] += 0.6
            else:
                regime_scores['ranging'] += 0.4
            
            # Volatility indicators
            if current_volatility > avg_volatility * 1.5:
                regime_scores['volatile'] += 0.4
            elif current_volatility < avg_volatility * 0.5:
                regime_scores['quiet'] += 0.3
            
            # Price action
            recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
            avg_range = (df['high'] - df['low']).mean()
            
            if recent_range < avg_range * 0.7:
                regime_scores['ranging'] += 0.3
                regime_scores['quiet'] += 0.2
            
            # Get dominant regime
            dominant_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            return {
                "type": dominant_regime[0],
                "confidence": float(dominant_regime[1]),
                "scores": regime_scores,
                "characteristics": {
                    "trend_strength": float(adx),
                    "volatility_ratio": float(current_volatility / avg_volatility) if avg_volatility > 0 else 1.0,
                    "price_momentum": float(latest.get('momentum_20', 0)),
                    "volume_profile": "high" if latest['volume'] > df['volume'].mean() * 1.2 else "normal"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {"type": "unknown", "confidence": 0.0}

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics"""
        try:
            returns = df['close'].pct_change().dropna()
            
            # Basic statistics
            daily_vol = returns.std()
            annualized_vol = daily_vol * np.sqrt(252)  # Assuming daily data
            
            # Value at Risk (VaR) - 95% confidence
            var_95 = np.percentile(returns, 5)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe ratio (assuming 0 risk-free rate)
            sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            sortino = returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calmar ratio
            calmar = returns.mean() / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Beta (correlation with market - simplified)
            # In practice, you'd compare with a market index
            beta = 1.0  # Placeholder
            
            # Risk-adjusted metrics
            risk_metrics = {
                "volatility": {
                    "daily": float(daily_vol),
                    "annualized": float(annualized_vol),
                    "current": float(df['volatility_20'].iloc[-1]) if 'volatility_20' in df.columns else float(daily_vol)
                },
                "value_at_risk": {
                    "var_95": float(var_95),
                    "cvar_95": float(cvar_95)
                },
                "drawdown": {
                    "max": float(max_drawdown),
                    "current": float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0
                },
                "ratios": {
                    "sharpe": float(sharpe),
                    "sortino": float(sortino),
                    "calmar": float(calmar)
                },
                "risk_score": self._calculate_risk_score(df, daily_vol, max_drawdown)
            }
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                "volatility": {"daily": 0.0, "annualized": 0.0},
                "value_at_risk": {"var_95": 0.0, "cvar_95": 0.0},
                "drawdown": {"max": 0.0, "current": 0.0},
                "ratios": {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0},
                "risk_score": 0.5
            }

    def _calculate_risk_score(self, df: pd.DataFrame, volatility: float, max_drawdown: float) -> float:
        """Calculate overall risk score (0-1, higher = riskier)"""
        try:
            risk_factors = []
            
            # Volatility risk
            vol_percentile = (df['volatility_20'].rank(pct=True).iloc[-1] 
                            if 'volatility_20' in df.columns else 0.5)
            risk_factors.append(vol_percentile)
            
            # Drawdown risk
            dd_risk = min(abs(max_drawdown) * 10, 1.0)  # Scale drawdown
            risk_factors.append(dd_risk)
            
            # Technical indicator risks
            latest = df.iloc[-1]
            
            # RSI extremes
            rsi = latest.get('rsi_14', 50)
            rsi_risk = max(0, (abs(rsi - 50) - 20) / 30)  # Risk increases beyond 30/70
            risk_factors.append(rsi_risk)
            
            # Bollinger Band position
            if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'close']):
                bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
                bb_risk = max(0, abs(bb_position - 0.5) - 0.3) / 0.2
                risk_factors.append(bb_risk)
            
            # Average risk score
            risk_score = np.mean(risk_factors)
            
            return float(np.clip(risk_score, 0, 1))
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5

    async def _generate_enhanced_signal(
        self, 
        df: pd.DataFrame, 
        patterns: Dict, 
        ml_prediction: Optional[Dict],
        sentiment: Dict,
        market_regime: Dict
    ) -> Optional[TradingSignal]:
        """Generate enhanced trading signal with multiple factors"""
        try:
            if df.empty:
                return None
            
            latest = df.iloc[-1]
            
            # Initialize signal components
            signal_scores = {
                'technical': 0.0,
                'pattern': 0.0,
                'ml': 0.0,
                'sentiment': 0.0,
                'regime': 0.0
            }
            
            weights = {
                'technical': 0.25,
                'pattern': 0.20,
                'ml': 0.30,
                'sentiment': 0.10,
                'regime': 0.15
            }
            
            # Technical analysis score
            tech_score = self._calculate_technical_score(latest)
            signal_scores['technical'] = tech_score
            
            # Pattern score
            pattern_score = self._calculate_pattern_score(patterns)
            signal_scores['pattern'] = pattern_score
            
            # ML prediction score
            if ml_prediction and ml_prediction.get('status') == 'success':
                ml_score = ml_prediction['probability']['buy'] - ml_prediction['probability']['sell']
                signal_scores['ml'] = ml_score
            
            # Sentiment score
            sentiment_score = sentiment['overall'] * sentiment['confidence']
            signal_scores['sentiment'] = sentiment_score
            
            # Market regime score
            regime_score = self._calculate_regime_score(market_regime)
            signal_scores['regime'] = regime_score
            
            # Calculate weighted composite score
            composite_score = sum(
                score * weights[component] 
                for component, score in signal_scores.items()
            )
            
            # Determine signal direction and strength
            if abs(composite_score) < 0.1:
                signal_type = SignalType.NEUTRAL
            elif composite_score > 0.3:
                signal_type = SignalType.STRONG_BUY
            elif composite_score > 0.1:
                signal_type = SignalType.BUY
            elif composite_score < -0.3:
                signal_type = SignalType.STRONG_SELL
            else:
                signal_type = SignalType.SELL
            
            # Skip neutral signals
            if signal_type == SignalType.NEUTRAL:
                return None
            
            # Calculate entry and exit levels
            atr = latest.get('atr_14', latest['close'] * 0.001)
            current_price = latest['close']
            
            # Adjust levels based on signal strength
            multiplier = 1.5 if 'STRONG' in signal_type.value else 1.0
            
            if 'BUY' in signal_type.value:
                entry_price = current_price
                stop_loss = current_price - (atr * 2 * multiplier)
                take_profits = [
                    current_price + (atr * 2 * multiplier),
                    current_price + (atr * 4 * multiplier),
                    current_price + (atr * 6 * multiplier)
                ]
            else:
                entry_price = current_price
                stop_loss = current_price + (atr * 2 * multiplier)
                take_profits = [
                    current_price - (atr * 2 * multiplier),
                    current_price - (atr * 4 * multiplier),
                    current_price - (atr * 6 * multiplier)
                ]
            
            # Calculate risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profits[0] - entry_price)
            risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Get dominant pattern
            dominant_pattern = max(patterns.items(), key=lambda x: abs(x[1]))[0] if patterns else "none"
            
            # Create signal
            signal = TradingSignal(
                symbol=df.index.name or "USDJPY",
                direction=signal_type,
                entry_price=float(entry_price),
                stop_loss=float(stop_loss),
                take_profit_levels=[float(tp) for tp in take_profits],
                confidence=float(abs(composite_score)),
                pattern=dominant_pattern,
                risk_reward_ratio=float(risk_reward_ratio),
                market_context={
                    "regime": market_regime['type'],
                    "volatility": float(latest.get('volatility_20', 0)),
                    "trend_strength": float(latest.get('adx', 0)),
                    "volume_profile": "high" if latest['volume'] > df['volume'].mean() * 1.2 else "normal"
                },
                timestamp=datetime.now(),
                metadata={
                    "scores": signal_scores,
                    "composite_score": float(composite_score),
                    "technical_levels": {
                        "support": float(latest.get('s1', 0)),
                        "resistance": float(latest.get('r1', 0)),
                        "pivot": float(latest.get('pivot', 0))
                    }
                }
            )
            
            # Store signal in history
            self.signal_history.append(signal)
            if len(self.signal_history) > 100:
                self.signal_history = self.signal_history[-100:]
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return None

    def _calculate_technical_score(self, latest: pd.Series) -> float:
        """Calculate technical analysis score"""
        try:
            score = 0.0
            
            # RSI
            rsi = latest.get('rsi_14', 50)
            if rsi < 30:
                score += 0.3
            elif rsi > 70:
                score -= 0.3
            
            # MACD
            macd = latest.get('macd', 0)
            macd_signal = latest.get('macd_signal', 0)
            if macd > macd_signal:
                score += 0.2
            else:
                score -= 0.2
            
            # Moving averages
            close = latest['close']
            sma_20 = latest.get('sma_20', close)
            sma_50 = latest.get('sma_50', close)
            
            if close > sma_20 > sma_50:
                score += 0.3
            elif close < sma_20 < sma_50:
                score -= 0.3
            
            # ADX trend strength
            adx = latest.get('adx', 0)
            if adx > 25:
                plus_di = latest.get('plus_di', 0)
                minus_di = latest.get('minus_di', 0)
                if plus_di > minus_di:
                    score += 0.2
                else:
                    score -= 0.2
            
            return np.clip(score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 0.0

    def _calculate_pattern_score(self, patterns: Dict) -> float:
        """Calculate pattern-based score"""
        try:
            bullish_patterns = [
                'hammer', 'inverted_hammer', 'engulfing_bull', 'morning_star',
                'three_white_soldiers', 'bullish_harami', 'double_bottom',
                'inverse_head_shoulders', 'ascending_triangle', 'falling_wedge',
                'bull_flag', 'golden_cross', 'bullish_divergence'
            ]
            
            bearish_patterns = [
                'hanging_man', 'shooting_star', 'evening_star', 'three_black_crows',
                'bearish_harami', 'dark_cloud_cover', 'double_top', 'head_shoulders',
                'descending_triangle', 'rising_wedge', 'bear_flag', 'death_cross',
                'bearish_divergence'
            ]
            
            score = 0.0
            
            for pattern, confidence in patterns.items():
                if pattern in bullish_patterns:
                    score += confidence * 0.1
                elif pattern in bearish_patterns:
                    score -= abs(confidence) * 0.1
            
            return np.clip(score, -1, 1)
            
        except Exception as e:
            logger.error(f"Error calculating pattern score: {e}")
            return 0.0

    def _calculate_regime_score(self, market_regime: Dict) -> float:
        """Calculate score based on market regime"""
        try:
            regime_type = market_regime['type']
            confidence = market_regime['confidence']
            
            # Trend-following bias in trending markets
            if regime_type == 'trending_up':
                return confidence * 0.5
            elif regime_type == 'trending_down':
                return -confidence * 0.5
            elif regime_type == 'volatile':
                # Reduce signal strength in volatile markets
                return 0.0
            else:
                # Neutral in ranging markets
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating regime score: {e}")
            return 0.0

    async def retrain_model(self, min_samples: int = 1000) -> Dict:
        """Retrain ML models with latest data"""
        try:
            logger.info("Starting model retraining...")
            
            # Get sufficient historical data
            df = await self.get_market_data(count=max(5000, min_samples))
            
            if df.empty or len(df) < min_samples:
                logger.error(f"Insufficient data for retraining. Got {len(df)} samples, need {min_samples}")
                return {"status": "error", "message": "Insufficient data"}
            
            # Train models
            result = self.predictor.train_models(df)
            
            # Log result
            logger.info(f"Retraining result: {result}")
            
            # Save training metadata
            if self.redis_client and result.get('status') == 'success':
                try:
                    await self.redis_client.setex(
                        "ml:last_training",
                        86400,  # 24 hours
                        json.dumps({
                            "timestamp": datetime.now().isoformat(),
                            "result": result,
                            "samples": len(df)
                        })
                    )
                except Exception as e:
                    logger.warning(f"Failed to save training metadata: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return {"status": "error", "message": str(e)}

    async def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            metrics = {
                "ml_model": {
                    "is_trained": self.predictor.is_trained,
                    "features_count": len(self.predictor.selected_features),
                    "models_count": len(self.predictor.models),
                    "last_performance": self.predictor.performance_history[-1] if self.predictor.performance_history else None
                },
                "signals": {
                    "total_generated": len(self.signal_history),
                    "last_24h": sum(1 for s in self.signal_history 
                                  if s.timestamp > datetime.now() - timedelta(days=1)),
                    "distribution": {}
                },
                "system": {
                    "redis_connected": self.redis_client is not None,
                    "db_connected": self.db_engine is not None,
                    "uptime": datetime.now().isoformat()
                }
            }
            
            # Calculate signal distribution
            if self.signal_history:
                for signal in self.signal_history[-50:]:  # Last 50 signals
                    direction = signal.direction.value
                    metrics['signals']['distribution'][direction] = \
                        metrics['signals']['distribution'].get(direction, 0) + 1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {"status": "error", "message": str(e)}

async def main():
    """Main function to run the enhanced ML agent"""
    service = TradingMLService()
    
    try:
        # Initialize service
        await service.initialize()
        
        if not service.is_initialized:
            logger.error("Failed to initialize service")
            return
        
        # Initial model training if needed
        if not service.predictor.is_trained:
            logger.info("Training initial models...")
            result = await service.retrain_model()
            if result.get('status') != 'success':
                logger.error("Failed to train initial models")
                return
        
        # Main loop
        last_retrain = datetime.now()
        symbols = ["USDJPY", "EURUSD", "GBPUSD"]
        
        while True:
            try:
                # Generate signals for all symbols
                for symbol in symbols:
                    analysis = await service.analyze_market(symbol)
                    
                    if analysis.get('signal'):
                        logger.info(f"Generated signal for {symbol}: {analysis['signal']}")
                        
                        # Send signal to API if available
                        try:
                            response = requests.post(
                                f"{API_URL}/api/signals",
                                json=analysis['signal'],
                                timeout=10,
                                headers={'Content-Type': 'application/json'}
                            )
                            if response.status_code == 200:
                                logger.info(f"Signal sent successfully for {symbol}")
                            else:
                                logger.warning(f"Signal API returned status {response.status_code}")
                        except Exception as e:
                            logger.error(f"Failed to send signal: {e}")
                
                # Check if retraining is needed
                if datetime.now() - last_retrain > timedelta(seconds=MODEL_UPDATE_INTERVAL):
                    logger.info("Starting scheduled model retraining...")
                    await service.retrain_model()
                    last_retrain = datetime.now()
                
                # Log performance metrics
                metrics = await service.get_performance_metrics()
                logger.info(f"Performance metrics: {json.dumps(metrics, indent=2)}")
                
                # Wait before next iteration
                await asyncio.sleep(900)  # 15 minutes
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
                
    except KeyboardInterrupt:
        logger.info("Shutting down ML agent...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        # Cleanup
        if service.redis_client:
            await service.redis_client.close()

if __name__ == "__main__":
    asyncio.run(main())
