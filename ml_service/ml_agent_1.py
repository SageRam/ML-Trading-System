# ml_agent.py - Enhanced ML Analytics Service
import asyncio
import json
import logging
import os
import pickle
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import redis.asyncio as redis
import talib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
import MetaTrader5 as mt5
import requests
from textblob import TextBlob
import yfinance as yf

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading_user:trading_password@localhost:5432/trading_system")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
API_URL = os.getenv("API_URL", "http://localhost:8080")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Technical indicators calculator"""

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        [cite_start]"""Calculate comprehensive technical indicators""" [cite: 2]
        try:
            # Ensure we have OHLCV data
            if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                [cite_start]logger.error("Missing required OHLCV columns") [cite: 2]
                return df

            [cite_start]high = df['high'].values [cite: 3]
            [cite_start]low = df['low'].values [cite: 3]
            [cite_start]close = df['close'].values [cite: 3]
            [cite_start]volume = df['volume'].values [cite: 3]

            # Trend indicators
            [cite_start]df['sma_20'] = talib.SMA(close, timeperiod=20) [cite: 4]
            [cite_start]df['sma_50'] = talib.SMA(close, timeperiod=50) [cite: 4]
            [cite_start]df['ema_12'] = talib.EMA(close, timeperiod=12) [cite: 4]
            [cite_start]df['ema_26'] = talib.EMA(close, timeperiod=26) [cite: 4]

            # MACD
            [cite_start]df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(close) [cite: 5]

            # RSI
            [cite_start]df['rsi'] = talib.RSI(close, timeperiod=14) [cite: 5]

            # Bollinger Bands
            [cite_start]df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(close) [cite: 5]

            # Stochastic
            [cite_start]df['stoch_k'], df['stoch_d'] = talib.STOCH(high, low, close) [cite: 6]

            # ATR
            [cite_start]df['atr'] = talib.ATR(high, low, close, timeperiod=14) [cite: 6]

            # ADX
            [cite_start]df['adx'] = talib.ADX(high, low, close, timeperiod=14) [cite: 6]

            # Volume indicators
            [cite_start]df['obv'] = talib.OBV(close, volume) [cite: 7]
            [cite_start]df['ad'] = talib.AD(high, low, close, volume) [cite: 7]

            # Price action indicators
            [cite_start]df['price_change'] = df['close'].pct_change() [cite: 7]
            [cite_start]df['volatility'] = df['price_change'].rolling(window=20).std() [cite: 8]
            [cite_start]df['momentum'] = df['close'] - df['close'].shift(10) [cite: 8]

            # Support/Resistance levels
            [cite_start]df['pivot'] = (df['high'] + df['low'] + df['close']) / 3 [cite: 8]
            [cite_start]df['r1'] = 2 * df['pivot'] - df['low'] [cite: 8]
            [cite_start]df['s1'] = 2 * df['pivot'] - df['high'] [cite: 9]

            # Market regime indicators
            [cite_start]df['trend_strength'] = df['adx'] [cite: 9]
            [cite_start]df['is_trending'] = df['adx'] > 25 [cite: 9]
            [cite_start]df['is_bullish'] = df['close'] > df['sma_20'] [cite: 9]

            [cite_start]return df [cite: 10]

        except Exception as e:
            [cite_start]logger.error(f"Error calculating indicators: {e}") [cite: 10]
            return df

class PatternRecognition:
    """Advanced pattern recognition system"""

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, float]:
        [cite_start]"""Detect trading patterns and return confidence scores""" [cite: 11]
        [cite_start]patterns = {} [cite: 11]

        try:
            if len(df) < 50:
                return patterns

            # Candlestick patterns using TA-Lib
            [cite_start]high = df['high'].values [cite: 12]
            [cite_start]low = df['low'].values [cite: 12]
            [cite_start]close = df['close'].values [cite: 12]
            [cite_start]open_prices = df['open'].values [cite: 12]

            # Bullish patterns
            [cite_start]patterns['hammer'] = np.sum(talib.CDLHAMMER(open_prices, high, low, close)[-20:]) / 20 [cite: 12]
            [cite_start]patterns['doji'] = np.sum(talib.CDLDOJI(open_prices, high, low, close)[-20:]) / 20 [cite: 13]
            [cite_start]patterns['engulfing_bull'] = np.sum(talib.CDLENGULFING(open_prices, high, low, close)[-20:]) / 20 [cite: 13]
            [cite_start]patterns['morning_star'] = np.sum(talib.CDLMORNINGSTAR(open_prices, high, low, close)[-20:]) / 20 [cite: 13]

            # Bearish patterns
            [cite_start]patterns['hanging_man'] = np.sum(talib.CDLHANGINGMAN(open_prices, high, low, close)[-20:]) / 20 [cite: 13]
            [cite_start]patterns['shooting_star'] = np.sum(talib.CDLSHOOTINGSTAR(open_prices, high, low, close)[-20:]) / 20 [cite: 14]
            [cite_start]patterns['evening_star'] = np.sum(talib.CDLEVENINGSTAR(open_prices, high, low, close)[-20:]) / 20 [cite: 14]

            # Chart patterns
            [cite_start]patterns['double_top'] = PatternRecognition._detect_double_top(df) [cite: 14]
            [cite_start]patterns['double_bottom'] = PatternRecognition._detect_double_bottom(df) [cite: 14]
            [cite_start]patterns['head_shoulders'] = PatternRecognition._detect_head_shoulders(df) [cite: 14]
            [cite_start]patterns['triangle'] = PatternRecognition._detect_triangle(df) [cite: 15]

            # Trend patterns
            [cite_start]patterns['uptrend'] = PatternRecognition._detect_uptrend(df) [cite: 15]
            [cite_start]patterns['downtrend'] = PatternRecognition._detect_downtrend(df) [cite: 15]
            [cite_start]patterns['sideways'] = PatternRecognition._detect_sideways(df) [cite: 15]

            [cite_start]return patterns [cite: 15]

        except Exception as e:
            [cite_start]logger.error(f"Error detecting patterns: {e}") [cite: 16]
            return patterns

    @staticmethod
    def _detect_double_top(df: pd.DataFrame) -> float:
        """Detect double top pattern"""
        try:
            if len(df) < 50:
                [cite_start]return 0.0 [cite: 17]

            [cite_start]highs = df['high'].rolling(window=5).max() [cite: 17]
            [cite_start]peaks = [] [cite: 17]

            for i in range(5, len(highs) - 5):
                [cite_start]if highs.iloc[i] == highs.iloc[i-5:i+6].max(): [cite: 18]
                    [cite_start]peaks.append((i, highs.iloc[i])) [cite: 18]

            if len(peaks) >= 2:
                [cite_start]last_two_peaks = peaks[-2:] [cite: 18]
                [cite_start]height_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1]) [cite: 19]
                [cite_start]time_diff = last_two_peaks[1][0] - last_two_peaks[0][0] [cite: 19]

                if height_diff < df['atr'].iloc[-1] * 2 and 10 < time_diff < 50:
                    [cite_start]return min(0.8, 1.0 - (height_diff / (df['atr'].iloc[-1] * 2))) [cite: 19]

            [cite_start]return 0.0 [cite: 20]

        except Exception:
            return 0.0

    @staticmethod
    def _detect_double_bottom(df: pd.DataFrame) -> float:
        """Detect double bottom pattern"""
        try:
            if len(df) < 50:
                [cite_start]return 0.0 [cite: 21]

            [cite_start]lows = df['low'].rolling(window=5).min() [cite: 21]
            [cite_start]troughs = [] [cite: 21]

            for i in range(5, len(lows) - 5):
                if lows.iloc[i] == lows.iloc[i-5:i+6].min():
                    [cite_start]troughs.append((i, lows.iloc[i])) [cite: 22]

            if len(troughs) >= 2:
                [cite_start]last_two_troughs = troughs[-2:] [cite: 22]
                [cite_start]height_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1]) [cite: 22]
                [cite_start]time_diff = last_two_troughs[1][0] - last_two_troughs[0][0] [cite: 23]

                if height_diff < df['atr'].iloc[-1] * 2 and 10 < time_diff < 50:
                    [cite_start]return min(0.8, 1.0 - (height_diff / (df['atr'].iloc[-1] * 2))) [cite: 23]

            [cite_start]return 0.0 [cite: 23]

        except Exception:
            [cite_start]return 0.0 [cite: 24]

    @staticmethod
    def _detect_head_shoulders(df: pd.DataFrame) -> float:
        """Detect head and shoulders pattern"""
        # Simplified implementation
        return 0.0

    @staticmethod
    def _detect_triangle(df: pd.DataFrame) -> float:
        [cite_start]"""Detect triangle pattern""" [cite: 25]
        # Simplified implementation
        return 0.0

    @staticmethod
    def _detect_uptrend(df: pd.DataFrame) -> float:
        """Detect uptrend"""
        try:
            if len(df) < 20:
                [cite_start]return 0.0 [cite: 26]

            [cite_start]recent_closes = df['close'].tail(20) [cite: 26]
            [cite_start]slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0] [cite: 26]

            [cite_start]return max(0.0, min(1.0, slope / (df['atr'].iloc[-1] * 0.1))) [cite: 26]

        except Exception:
            [cite_start]return 0.0 [cite: 26]

    @staticmethod
    def _detect_downtrend(df: pd.DataFrame) -> float:
        """Detect downtrend"""
        try:
            if len(df) < 20:
                [cite_start]return 0.0 [cite: 27]

            [cite_start]recent_closes = df['close'].tail(20) [cite: 27]
            [cite_start]slope = np.polyfit(range(len(recent_closes)), recent_closes, 1)[0] [cite: 28]

            [cite_start]return max(0.0, min(1.0, -slope / (df['atr'].iloc[-1] * 0.1))) [cite: 28]

        except Exception:
            return 0.0

    @staticmethod
    def _detect_sideways(df: pd.DataFrame) -> float:
        """Detect sideways market"""
        try:
            if len(df) < 20:
                [cite_start]return 0.0 [cite: 29]

            [cite_start]recent_closes = df['close'].tail(20) [cite: 29]
            [cite_start]volatility = recent_closes.std() [cite: 29]

            [cite_start]return max(0.0, min(1.0, 1.0 - (volatility / df['atr'].iloc[-1]))) [cite: 30]

        except Exception:
            return 0.0

class SentimentAnalyzer:
    """Market sentiment analysis"""

    @staticmethod
    async def analyze_news_sentiment(symbol: str = "USDJPY") -> Dict:
        """Analyze news sentiment for given symbol"""
        try:
            # In a real implementation, you would fetch news from APIs like:
            # - NewsAPI, Alpha Vantage, Yahoo Finance, etc.
            # For now, we'll simulate sentiment analysis

            sentiment_data = {
                [cite_start]'overall': np.random.uniform(-0.3, 0.3),  # Random sentiment between -0.3 and 0.3 [cite: 31]
                [cite_start]'confidence': np.random.uniform(0.5, 0.9), [cite: 32]
                'sources': {
                    [cite_start]'reuters': np.random.uniform(-0.5, 0.5), [cite: 32]
                    [cite_start]'bloomberg': np.random.uniform(-0.5, 0.5), [cite: 32]
                    [cite_start]'forex_factory': np.random.uniform(-0.5, 0.5) [cite: 33]
                },
                [cite_start]'themes': ['inflation', 'central_bank', 'economic_data'], [cite: 33]
                [cite_start]'last_updated': datetime.now().isoformat() [cite: 33]
            }

            [cite_start]return sentiment_data [cite: 33]

        except Exception as e:
            [cite_start]logger.error(f"Error analyzing sentiment: {e}") [cite: 34]
            return {
                'overall': 0.0,
                'confidence': 0.0,
                'sources': {},
                [cite_start]'themes': [], [cite: 35]
                'last_updated': datetime.now().isoformat()
            }

class MLPredictor:
    """Machine Learning prediction engine"""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        [cite_start]self.is_trained = False [cite: 36]

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            # Technical indicators
            [cite_start]df = TechnicalIndicators.calculate_indicators(df) [cite: 36]

            # Pattern recognition
            [cite_start]patterns = PatternRecognition.detect_patterns(df) [cite: 37]
            for pattern_name, confidence in patterns.items():
                [cite_start]df[f'pattern_{pattern_name}'] = confidence [cite: 37]

            # Time-based features
            [cite_start]df['hour'] = pd.to_datetime(df.index).hour [cite: 37]
            [cite_start]df['day_of_week'] = pd.to_datetime(df.index).dayofweek [cite: 37]
            [cite_start]df['is_market_open'] = ((df['hour'] >= 0) & (df['hour'] <= 23)).astype(int) [cite: 38]

            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                [cite_start]df[f'close_lag_{lag}'] = df['close'].shift(lag) [cite: 38]
                [cite_start]df[f'volume_lag_{lag}'] = df['volume'].shift(lag) [cite: 38]
                [cite_start]df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag) [cite: 39]

            # Rolling statistics
            for window in [5, 10, 20]:
                [cite_start]df[f'close_mean_{window}'] = df['close'].rolling(window).mean() [cite: 39]
                [cite_start]df[f'close_std_{window}'] = df['close'].rolling(window).std() [cite: 39]
                [cite_start]df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean() [cite: 40]

            # Target variable (future price movement)
            [cite_start]df['future_return'] = df['close'].shift(-1) / df['close'] - 1 [cite: 40]
            [cite_start]df['target'] = (df['future_return'] > 0.0001).astype(int)  # 1 pip target [cite: 40]

            [cite_start]return df [cite: 40]

        except Exception as e:
            [cite_start]logger.error(f"Error preparing features: {e}") [cite: 41]
            return df

    def train_model(self, df: pd.DataFrame) -> Dict:
        """Train the ML model"""
        try:
            [cite_start]logger.info("Training ML model...") [cite: 41]

            # Prepare features
            [cite_start]df = self.prepare_features(df) [cite: 42]

            # Select feature columns
            feature_cols = [col for col in df.columns if col not in [
                [cite_start]'open', 'high', 'low', 'close', 'volume', 'time', [cite: 43]
                [cite_start]'future_return', 'target' [cite: 43]
            ] and not df[col].isna().all()]

            [cite_start]self.feature_columns = feature_cols [cite: 43]

            # Prepare data
            [cite_start]X = df[feature_cols].fillna(0) [cite: 44]
            [cite_start]y = df['target'].fillna(0) [cite: 44]

            # Remove rows with NaN target
            [cite_start]mask = ~y.isna() [cite: 44]
            [cite_start]X = X[mask] [cite: 44]
            [cite_start]y = y[mask] [cite: 44]

            if len(X) < 100:
                [cite_start]logger.warning("Insufficient data for training") [cite: 45]
                [cite_start]return {"status": "error", "message": "Insufficient data"} [cite: 45]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                [cite_start]X, y, test_size=0.2, random_state=42, stratify=y [cite: 46]
            )

            # Scale features
            [cite_start]X_train_scaled = self.scaler.fit_transform(X_train) [cite: 46]
            [cite_start]X_test_scaled = self.scaler.transform(X_test) [cite: 46]

            # Train ensemble model
            models = {
                [cite_start]'rf': RandomForestClassifier(n_estimators=100, random_state=42), [cite: 47]
                [cite_start]'gb': GradientBoostingClassifier(n_estimators=100, random_state=42) [cite: 47]
            }

            [cite_start]trained_models = {} [cite: 48]
            [cite_start]scores = {} [cite: 48]

            for name, model in models.items():
                [cite_start]model.fit(X_train_scaled, y_train) [cite: 48]
                [cite_start]y_pred = model.predict(X_test_scaled) [cite: 48]
                [cite_start]score = accuracy_score(y_test, y_pred) [cite: 48]

                [cite_start]trained_models[name] = model [cite: 49]
                [cite_start]scores[name] = score [cite: 49]

                [cite_start]logger.info(f"{name} accuracy: {score:.4f}") [cite: 49]

            # Use the best model
            [cite_start]best_model_name = max(scores, key=scores.get) [cite: 50]
            [cite_start]self.model = trained_models[best_model_name] [cite: 50]
            [cite_start]self.is_trained = True [cite: 50]

            # Save model
            [cite_start]model_dir = "models" [cite: 50]
            [cite_start]os.makedirs(model_dir, exist_ok=True) [cite: 51]

            with open(f"{model_dir}/ml_model.pkl", "wb") as f:
                pickle.dump({
                    'model': self.model,
                    [cite_start]'scaler': self.scaler, [cite: 51]
                    [cite_start]'feature_columns': self.feature_columns [cite: 52]
                }, f)

            return {
                "status": "success",
                [cite_start]"accuracy": scores[best_model_name], [cite: 52]
                [cite_start]"model_type": best_model_name, [cite: 53]
                [cite_start]"features_count": len(self.feature_columns), [cite: 53]
                [cite_start]"training_samples": len(X_train) [cite: 53]
            }

        except Exception as e:
            [cite_start]logger.error(f"Error training model: {e}") [cite: 53]
            [cite_start]traceback.print_exc() [cite: 54]
            [cite_start]return {"status": "error", "message": str(e)} [cite: 54]

    def load_model(self) -> bool:
        """Load trained model"""
        try:
            [cite_start]model_path = "models/ml_model.pkl" [cite: 54]
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    [cite_start]data = pickle.load(f) [cite: 55]
                    [cite_start]self.model = data['model'] [cite: 55]
                    [cite_start]self.scaler = data['scaler'] [cite: 55]
                    [cite_start]self.feature_columns = data['feature_columns'] [cite: 55]
                    [cite_start]self.is_trained = True [cite: 56]
                    [cite_start]logger.info("Model loaded successfully") [cite: 56]
                    return True
            return False
        except Exception as e:
            [cite_start]logger.error(f"Error loading model: {e}") [cite: 56]
            return False

    def predict(self, df: pd.DataFrame) -> Dict:
        [cite_start]"""Make predictions""" [cite: 57]
        try:
            if not self.is_trained:
                [cite_start]return {"status": "error", "message": "Model not trained"} [cite: 57]

            # Prepare features
            [cite_start]df = self.prepare_features(df) [cite: 58]

            # Get latest data point
            [cite_start]latest_data = df[self.feature_columns].iloc[-1:].fillna(0) [cite: 58]

            # Scale features
            [cite_start]latest_scaled = self.scaler.transform(latest_data) [cite: 58]

            # Make prediction
            [cite_start]prediction = self.model.predict(latest_scaled)[0] [cite: 59]
            [cite_start]probability = self.model.predict_proba(latest_scaled)[0] [cite: 59]

            return {
                "status": "success",
                [cite_start]"prediction": int(prediction), [cite: 59]
                "probability": {
                    [cite_start]"sell": float(probability[0]), [cite: 60]
                    [cite_start]"buy": float(probability[1]) [cite: 60]
                },
                [cite_start]"confidence": float(max(probability)) [cite: 61]
            }

        except Exception as e:
            [cite_start]logger.error(f"Error making prediction: {e}") [cite: 61]
            return {"status": "error", "message": str(e)}

class TradingMLService:
    """Main ML service for trading system"""

    def __init__(self):
        self.redis_client = None
        [cite_start]self.db_engine = create_engine(DATABASE_URL) [cite: 61]
        [cite_start]self.predictor = MLPredictor() [cite: 62]
        [cite_start]self.sentiment_analyzer = SentimentAnalyzer() [cite: 62]

    async def initialize(self):
        """Initialize the service"""
        try:
            # Connect to Redis
            [cite_start]self.redis_client = redis.from_url(REDIS_URL) [cite: 62]
            [cite_start]await self.redis_client.ping() [cite: 63]
            [cite_start]logger.info("Connected to Redis") [cite: 63]

            # Load or train model
            if not self.predictor.load_model():
                [cite_start]logger.info("No existing model found, will train on new data") [cite: 63]

        except Exception as e:
            [cite_start]logger.error(f"Error initializing service: {e}") [cite: 64]

    async def get_market_data(self, symbol: str = "USDJPY", timeframe: str = "M15", count: int = 1000) -> pd.DataFrame:
        """Get market data from MT5 or API"""
        try:
            # Try to get data from API first
            [cite_start]response = requests.get(f"{API_URL}/api/market-data/{symbol}?timeframe={timeframe}&count={count}") [cite: 64]
            [cite_start]if response.status_code == 200: [cite: 65]
                [cite_start]data = response.json() [cite: 65]
                [cite_start]df = pd.DataFrame(data['data']) [cite: 65]
                [cite_start]df['time'] = pd.to_datetime(df['time']) [cite: 65]
                [cite_start]df.set_index('time', inplace=True) [cite: 65]
                return df

            # Fallback to Yahoo Finance for demo purposes
            [cite_start]ticker_map = {"USDJPY": "JPY=X", "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X"} [cite: 66]
            [cite_start]ticker = ticker_map.get(symbol, "JPY=X") [cite: 66]

            [cite_start]yf_data = yf.download(ticker, period="1mo", interval="15m") [cite: 66]
            if not yf_data.empty:
                [cite_start]yf_data.columns = [col.lower() for col in yf_data.columns] [cite: 67]
                [cite_start]yf_data['volume'] = yf_data.get('volume', 1000)  # Default volume if not available [cite: 67]
                return yf_data.tail(count)

            [cite_start]return pd.DataFrame() [cite: 67]

        except Exception as e:
            [cite_start]logger.error(f"Error getting market data: {e}") [cite: 68]
            return pd.DataFrame()

    async def analyze_market(self, symbol: str = "USDJPY") -> Dict:
        """Comprehensive market analysis"""
        try:
            # Get market data
            [cite_start]df = await self.get_market_data(symbol) [cite: 69]
            if df.empty:
                [cite_start]return {"status": "error", "message": "No market data available"} [cite: 69]

            # Technical analysis
            [cite_start]df = TechnicalIndicators.calculate_indicators(df) [cite: 69]

            # Pattern recognition
            [cite_start]patterns = PatternRecognition.detect_patterns(df) [cite: 70]

            # ML prediction
            [cite_start]ml_prediction = self.predictor.predict(df) [cite: 70]

            # Sentiment analysis
            [cite_start]sentiment = await self.sentiment_analyzer.analyze_news_sentiment(symbol) [cite: 70]

            # Current market state
            [cite_start]latest = df.iloc[-1] [cite: 71]

            # Generate trading signal
            [cite_start]signal = await self.generate_signal(df, patterns, ml_prediction, sentiment) [cite: 71]

            analysis = {
                "symbol": symbol,
                [cite_start]"timestamp": datetime.now().isoformat(), [cite: 72]
                "price": {
                    [cite_start]"current": float(latest['close']), [cite: 72]
                    [cite_start]"change": float(latest['close'] - df['close'].iloc[-2]), [cite: 73]
                    [cite_start]"change_pct": float((latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100) [cite: 73]
                },
                "technical_indicators": {
                    [cite_start]"rsi": float(latest['rsi']) if not pd.isna(latest['rsi']) else None, [cite: 73]
                    [cite_start]"macd": float(latest['macd']) if not pd.isna(latest['macd']) else None, [cite: 74]
                    [cite_start]"atr": float(latest['atr']) if not pd.isna(latest['atr']) else None, [cite: 74]
                    [cite_start]"adx": float(latest['adx']) if not pd.isna(latest['adx']) else None [cite: 74]
                },
                [cite_start]"patterns": patterns, [cite: 75]
                [cite_start]"ml_prediction": ml_prediction, [cite: 75]
                [cite_start]"sentiment": sentiment, [cite: 75]
                [cite_start]"signal": signal, [cite: 75]
                [cite_start]"market_regime": self.determine_market_regime(df), [cite: 75]
                [cite_start]"volatility": float(df['price_change'].tail(20).std()) if 'price_change' in df.columns else 0.0 [cite: 75]
            }

            # Cache analysis in Redis
            if self.redis_client:
                await self.redis_client.setex(
                    [cite_start]f"analysis:{symbol}", [cite: 76]
                    [cite_start]300,  # 5 minutes [cite: 77]
                    [cite_start]json.dumps(analysis, default=str) [cite: 77]
                )

            [cite_start]return analysis [cite: 77]

        except Exception as e:
            [cite_start]logger.error(f"Error in market analysis: {e}") [cite: 78]
            [cite_start]traceback.print_exc() [cite: 78]
            return {"status": "error", "message": str(e)}

    def determine_market_regime(self, df: pd.DataFrame) -> str:
        """Determine current market regime"""
        try:
            if len(df) < 20:
                [cite_start]return "unknown" [cite: 79]

            [cite_start]latest = df.iloc[-1] [cite: 79]

            # Check trend strength
            [cite_start]adx = latest.get('adx', 0) [cite: 79]

            if adx > 25:  # Strong trend
                [cite_start]sma_20 = latest.get('sma_20', latest['close']) [cite: 80]
                if latest['close'] > sma_20:
                    [cite_start]return "trending_up" [cite: 80]
                else:
                    [cite_start]return "trending_down" [cite: 80]
            else:  # Weak trend or ranging
                [cite_start]volatility = df['price_change'].tail(20).std() if 'price_change' in df.columns else 0 [cite: 81]
                [cite_start]avg_volatility = df['price_change'].std() if 'price_change' in df.columns else 0 [cite: 81]

                if volatility > avg_volatility * 1.5:
                    [cite_start]return "volatile" [cite: 82]
                else:
                    [cite_start]return "ranging" [cite: 82]

        except Exception:
            [cite_start]return "unknown" [cite: 82]

    async def generate_signal(self, df: pd.DataFrame, patterns: Dict, ml_prediction: Dict, sentiment: Dict) -> Optional[Dict]:
        [cite_start]"""Generate trading signal based on all analysis""" [cite: 83]
        try:
            if df.empty or ml_prediction.get('status') != 'success':
                [cite_start]return None [cite: 83]

            [cite_start]latest = df.iloc[-1] [cite: 84]

            # Calculate signal strength
            signal_strength = 0.0
            direction = None

            # ML prediction weight (40%)
            [cite_start]ml_confidence = ml_prediction.get('confidence', 0) [cite: 84]
            if ml_prediction.get('prediction') == 1:  # Buy signal
                [cite_start]signal_strength += ml_confidence * 0.4 [cite: 85]
                [cite_start]direction = "BUY" [cite: 85]
            else:  # Sell signal
                [cite_start]signal_strength += ml_confidence * 0.4 [cite: 85]
                [cite_start]direction = "SELL" [cite: 86]

            # Pattern recognition weight (30%)
            [cite_start]bullish_patterns = ['hammer', 'engulfing_bull', 'morning_star', 'double_bottom', 'uptrend'] [cite: 86]
            [cite_start]bearish_patterns = ['hanging_man', 'shooting_star', 'evening_star', 'double_top', 'downtrend'] [cite: 86]

            [cite_start]pattern_score = 0.0 [cite: 87]
            for pattern in bullish_patterns:
                [cite_start]pattern_score += patterns.get(pattern, 0) [cite: 87]
            for pattern in bearish_patterns:
                [cite_start]pattern_score -= patterns.get(pattern, 0) [cite: 87]

            [cite_start]signal_strength += abs(pattern_score) * 0.3 [cite: 87]

            # Technical indicators weight (20%)
            [cite_start]rsi = latest.get('rsi', 50) [cite: 88]
            [cite_start]macd = latest.get('macd', 0) [cite: 88]

            [cite_start]tech_score = 0.0 [cite: 88]
            if rsi < 30:  # Oversold
                [cite_start]tech_score += 0.5 [cite: 89]
            elif rsi > 70:  # Overbought
                [cite_start]tech_score -= 0.5 [cite: 89]

            if macd > 0:
                [cite_start]tech_score += 0.3 [cite: 89]
            else:
                [cite_start]tech_score -= 0.3 [cite: 90]

            [cite_start]signal_strength += abs(tech_score) * 0.2 [cite: 90]

            # Sentiment weight (10%)
            [cite_start]sentiment_score = sentiment.get('overall', 0) [cite: 90]
            [cite_start]signal_strength += abs(sentiment_score) * 0.1 [cite: 91]

            # Adjust direction based on combined analysis
            combined_score = (
                (1 if ml_prediction.get('prediction') == 1 else -1) * ml_confidence * 0.4 +
                [cite_start]pattern_score * 0.3 + [cite: 92]
                [cite_start]tech_score * 0.2 + [cite: 92]
                [cite_start]sentiment_score * 0.1 [cite: 92]
            )

            if combined_score > 0:
                [cite_start]direction = "BUY" [cite: 92]
            else:
                [cite_start]direction = "SELL" [cite: 93]

            # Only generate signal if confidence is high enough
            if signal_strength < 0.6:
                [cite_start]return None [cite: 93]

            # Calculate entry, stop loss, and take profit levels
            [cite_start]atr = latest.get('atr', 0.001) [cite: 94]
            [cite_start]current_price = latest['close'] [cite: 94]

            if direction == "BUY":
                [cite_start]entry_price = current_price [cite: 94]
                [cite_start]stop_loss = current_price - (atr * 2) [cite: 95]
                [cite_start]tp1 = current_price + (atr * 3) [cite: 95]
                [cite_start]tp2 = current_price + (atr * 5) [cite: 95]
                [cite_start]tp3 = current_price + (atr * 8) [cite: 95]
            else:
                [cite_start]entry_price = current_price [cite: 96]
                [cite_start]stop_loss = current_price + (atr * 2) [cite: 96]
                [cite_start]tp1 = current_price - (atr * 3) [cite: 96]
                [cite_start]tp2 = current_price - (atr * 5) [cite: 96]
                [cite_start]tp3 = current_price - (atr * 8) [cite: 97]

            # Calculate risk-reward ratio
            [cite_start]risk = abs(entry_price - stop_loss) [cite: 97]
            [cite_start]reward = abs(tp1 - entry_price) [cite: 97]
            [cite_start]risk_reward_ratio = reward / risk if risk > 0 else 0 [cite: 97]

            signal = {
                [cite_start]"symbol": "USDJPY", [cite: 98]
                [cite_start]"entry_price": float(entry_price), [cite: 98]
                [cite_start]"stop_loss": float(stop_loss), [cite: 98]
                [cite_start]"tp1": float(tp1), [cite: 98]
                [cite_start]"tp2": float(tp2), [cite: 99]
                [cite_start]"tp3": float(tp3), [cite: 99]
                [cite_start]"direction": direction, [cite: 99]
                [cite_start]"confidence": float(min(signal_strength, 0.95)), [cite: 99]
                [cite_start]"pattern": self.get_dominant_pattern(patterns), [cite: 99]
                [cite_start]"risk_reward_ratio": float(risk_reward_ratio), [cite: 100]
                [cite_start]"expected_value": float(signal_strength * reward), [cite: 100]
                "market_context": {
                    [cite_start]"regime": self.determine_market_regime(df), [cite: 100]
                    [cite_start]"volatility": float(df['price_change'].tail(20).std()) if 'price_change' in df.columns else 0.0, [cite: 100]
                    [cite_start]"trend_strength": float(latest.get('adx', 0)) [cite: 101]
                },
                [cite_start]"news_impact": float(abs(sentiment_score)) [cite: 101]
            }

            [cite_start]return signal [cite: 101]

        except Exception as e:
            [cite_start]logger.error(f"Error generating signal: {e}") [cite: 102]
            return None

    def get_dominant_pattern(self, patterns: Dict) -> str:
        """Get the most significant pattern"""
        if not patterns:
            return "mixed"

        max_pattern = max(patterns.items(), key=lambda x: abs(x[1]))
        if abs(max_pattern[1]) > 0.3:
            [cite_start]return max_pattern[0] [cite: 103]
        return "mixed"

    async def retrain_model(self):
        """Retrain the ML model with latest data"""
        try:
            [cite_start]logger.info("Retraining ML model...") [cite: 103]

            # Get more historical data for training
            [cite_start]df = await self.get_market_data(count=5000) [cite: 104]
            if df.empty:
                [cite_start]logger.error("No data available for retraining") [cite: 104]
                return

            # Train model
            [cite_start]result = self.predictor.train_model(df) [cite: 105]
            [cite_start]logger.info(f"Retraining result: {result}") [cite: 105]

            return result