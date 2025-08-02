# main.py - Production-Ready FastAPI Backend with ML Integration
import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy import create_engine, text, pool
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import aiohttp
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:xPgernhZHDqzwMZZSusJrpPuUxZyUYoR@ballast.proxy.rlwy.net:56300/railway")
REDIS_URL = os.getenv("REDIS_URL", "redis://default:vanoEzlkdWoeXeSbOYIrtagILrJaugmb@yamabiko.proxy.rlwy.net:22322")
ML_AGENT_URL = os.getenv("ML_AGENT_URL", "http://localhost:8081")  # Will be Railway URL
VERCEL_FRONTEND_URL = os.getenv("VERCEL_FRONTEND_URL", "https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database setup with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=pool.NullPool,  # For Railway compatibility
    connect_args={
        "sslmode": "require",
        "connect_timeout": 30
    }
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enhanced Pydantic models
class TradeSignal(BaseModel):
    symbol: str = Field(default="USDJPY", description="Trading symbol")
    entry_price: float = Field(description="Entry price for the trade")
    stop_loss: float = Field(description="Stop loss price")
    take_profit_levels: List[float] = Field(description="Take profit levels")
    direction: str = Field(description="Trade direction: BUY or SELL")
    confidence: float = Field(ge=0, le=1, description="Signal confidence")
    pattern: str = Field(description="Pattern that triggered the signal")
    risk_reward_ratio: float = Field(description="Risk reward ratio")
    market_context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = None

class MLAnalysis(BaseModel):
    symbol: str
    timestamp: datetime
    price: Dict[str, float]
    technical_indicators: Dict[str, float]
    patterns: Dict[str, float]
    ml_prediction: Dict[str, Any]
    sentiment: Dict[str, Any]
    market_regime: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    signal: Optional[TradeSignal] = None
    market_conditions: Dict[str, Any]

class TradeExecution(BaseModel):
    signal_id: str
    mt5_ticket: Optional[int] = None
    execution_price: float
    slippage: float = 0.0
    commission: float = 0.0
    status: str
    timestamp: datetime = Field(default_factory=datetime.now)

class PerformanceMetrics(BaseModel):
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    current_balance: float = 10000.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    monthly_pnl: float = 0.0

class SystemConfig(BaseModel):
    max_concurrent_trades: int = 3
    enable_auto_trading: bool = False
    risk_per_trade: float = 2.0
    storage_location: str = "railway"
    mt5_enabled: bool = True
    ml_enabled: bool = True
    allowed_symbols: List[str] = ["USDJPY", "EURUSD", "GBPUSD"]
    min_confidence: float = 0.7
    max_daily_trades: int = 10

# Global state
active_websockets: Set[WebSocket] = set()
redis_client: Optional[redis.Redis] = None
system_config: SystemConfig = SystemConfig()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

# FastAPI app with lifespan
app = FastAPI(
    title="ML Trading System Backend",
    description="Production backend for ML-powered trading system with Railway integration",
    version="3.0.0",
    lifespan=lifespan
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        VERCEL_FRONTEND_URL,
        "http://localhost:3000",  # For local development
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced MT5 Manager
class MT5Manager:
    def __init__(self):
        self.connected = False
        self.account_info = {}
        self.symbols_info = {}
        
    async def connect(self, account: int = None, password: str = None, server: str = None):
        """Connect to MT5 with retry logic"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MT5 (attempt {attempt + 1})")
                    await asyncio.sleep(retry_delay)
                    continue
                    
                if account and password and server:
                    if not mt5.login(account, password, server):
                        logger.error(f"Failed to login to account {account}")
                        await asyncio.sleep(retry_delay)
                        continue
                        
                self.connected = True
                self.account_info = mt5.account_info()._asdict() if mt5.account_info() else {}
                
                # Cache symbol information
                for symbol in system_config.allowed_symbols:
                    info = mt5.symbol_info(symbol)
                    if info:
                        self.symbols_info[symbol] = info._asdict()
                
                logger.info(f"MT5 connected successfully. Account: {self.account_info.get('login', 'Demo')}")
                return True
                
            except Exception as e:
                logger.error(f"MT5 connection error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(retry_delay)
                
        return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
    
    async def get_market_data(self, symbol: str = "USDJPY", timeframe: str = "M15", count: int = 1000):
        """Get market data from MT5 with error handling"""
        if not self.connected:
            logger.warning("MT5 not connected, returning mock data")
            return self._get_mock_market_data(symbol, timeframe, count)
            
        try:
            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
            }
            
            timeframe_mt5 = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
            
            if rates is None:
                logger.warning(f"No data received for {symbol}")
                return self._get_mock_market_data(symbol, timeframe, count)
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Add session markers
            df['session'] = df['time'].apply(self._get_trading_session)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': df.to_dict('records'),
                'last_update': datetime.now().isoformat(),
                'bid': mt5.symbol_info_tick(symbol).bid if mt5.symbol_info_tick(symbol) else df['close'].iloc[-1],
                'ask': mt5.symbol_info_tick(symbol).ask if mt5.symbol_info_tick(symbol) else df['close'].iloc[-1],
                'spread': self._calculate_spread(symbol)
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return self._get_mock_market_data(symbol, timeframe, count)
    
    def _get_mock_market_data(self, symbol: str, timeframe: str, count: int):
        """Generate mock market data for testing"""
        np.random.seed(42)
        base_price = {"USDJPY": 150.0, "EURUSD": 1.1, "GBPUSD": 1.3}.get(symbol, 1.0)
        
        timestamps = pd.date_range(end=datetime.now(), periods=count, freq='15min')
        prices = base_price + np.cumsum(np.random.randn(count) * 0.0001)
        
        data = []
        for i, (ts, price) in enumerate(zip(timestamps, prices)):
            high = price + abs(np.random.randn() * 0.0002)
            low = price - abs(np.random.randn() * 0.0002)
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            
            data.append({
                'time': ts.isoformat(),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'tick_volume': int(np.random.uniform(100, 1000)),
                'spread': int(np.random.uniform(1, 5)),
                'real_volume': 0
            })
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': data,
            'last_update': datetime.now().isoformat(),
            'bid': prices[-1],
            'ask': prices[-1] + 0.0002,
            'spread': 2
        }
    
    def _get_trading_session(self, dt: datetime) -> str:
        """Determine trading session based on time"""
        hour = dt.hour
        if 0 <= hour < 9:
            return "Tokyo"
        elif 8 <= hour < 17:
            return "London"
        elif 13 <= hour < 22:
            return "NewYork"
        else:
            return "Sydney"
    
    def _calculate_spread(self, symbol: str) -> float:
        """Calculate current spread in pips"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                point = self.symbols_info.get(symbol, {}).get('point', 0.00001)
                return (tick.ask - tick.bid) / point
            return 0
        except:
            return 0
    
    async def execute_trade(self, signal: TradeSignal, position_size: float = 0.01) -> Dict:
        """Execute trade with enhanced risk management"""
        if not self.connected:
            logger.warning("MT5 not connected, simulating trade execution")
            return self._simulate_trade_execution(signal, position_size)
            
        try:
            symbol = signal.symbol
            symbol_info = mt5.symbol_info(symbol)
            
            if symbol_info is None:
                return {"status": "error", "message": f"Symbol {symbol} not found"}
            
            # Calculate position size based on risk
            lot = self._calculate_position_size(signal, position_size)
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.take_profit_levels[0] if signal.take_profit_levels else 0,
                "deviation": 10,  # Maximum price deviation
                "magic": 123456,  # EA magic number
                "comment": f"ML-{signal.pattern}-{signal.confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return {
                    "status": "error",
                    "message": f"Order failed: {result.comment}",
                    "retcode": result.retcode
                }
            
            # Calculate slippage
            slippage = abs(result.price - signal.entry_price)
            
            return {
                "status": "success",
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume,
                "slippage": slippage,
                "commission": result.commission if hasattr(result, 'commission') else 0,
                "comment": result.comment
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}
    
    def _calculate_position_size(self, signal: TradeSignal, base_lot: float) -> float:
        """Calculate position size based on risk management"""
        try:
            if not self.account_info:
                return base_lot
                
            account_balance = self.account_info.get('balance', 10000)
            risk_amount = account_balance * (system_config.risk_per_trade / 100)
            
            # Calculate pip value
            symbol_info = self.symbols_info.get(signal.symbol, {})
            point = symbol_info.get('point', 0.00001)
            
            # Calculate stop loss in pips
            sl_pips = abs(signal.entry_price - signal.stop_loss) / point
            
            # Calculate lot size
            if sl_pips > 0:
                pip_value = symbol_info.get('trade_tick_value', 0.1)  # USD per pip per 0.01 lot
                lot_size = risk_amount / (sl_pips * pip_value * 100)  # Convert to standard lots
                
                # Apply limits
                min_lot = symbol_info.get('volume_min', 0.01)
                max_lot = symbol_info.get('volume_max', 10.0)
                lot_step = symbol_info.get('volume_step', 0.01)
                
                # Round to lot step
                lot_size = round(lot_size / lot_step) * lot_step
                
                return max(min_lot, min(lot_size, max_lot))
            
            return base_lot
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return base_lot
    
    def _simulate_trade_execution(self, signal: TradeSignal, position_size: float) -> Dict:
        """Simulate trade execution for testing"""
        import random
        
        # Simulate slippage
        slippage = random.uniform(-0.0002, 0.0002)
        execution_price = signal.entry_price + slippage
        
        return {
            "status": "success",
            "ticket": random.randint(100000, 999999),
            "price": execution_price,
            "volume": position_size,
            "slippage": abs(slippage),
            "commission": position_size * 7,  # $7 per lot
            "comment": "Simulated trade"
        }
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions with P&L calculation"""
        if not self.connected:
            return []
            
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
                
            position_list = []
            for pos in positions:
                current_price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask
                
                position_list.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": current_price,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "commission": pos.commission,
                    "comment": pos.comment,
                    "time": datetime.fromtimestamp(pos.time).isoformat(),
                    "pips": self._calculate_pips(pos.symbol, pos.price_open, current_price, pos.type)
                })
                
            return position_list
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _calculate_pips(self, symbol: str, open_price: float, current_price: float, position_type: int) -> float:
        """Calculate profit/loss in pips"""
        try:
            point = self.symbols_info.get(symbol, {}).get('point', 0.00001)
            if position_type == mt5.POSITION_TYPE_BUY:
                pips = (current_price - open_price) / point
            else:
                pips = (open_price - current_price) / point
            return round(pips, 1)
        except:
            return 0

    async def close_position(self, ticket: int) -> Dict:
        """Close a specific position"""
        if not self.connected:
            return {"status": "error", "message": "MT5 not connected"}
            
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                return {"status": "error", "message": "Position not found"}
                
            pos = position[0]
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.POSITION_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 123456,
                "comment": "Close by ML system",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "status": "error",
                    "message": f"Failed to close position: {result.comment}",
                    "retcode": result.retcode
                }
                
            return {
                "status": "success",
                "ticket": result.order,
                "price": result.price,
                "profit": pos.profit
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"status": "error", "message": str(e)}

# Initialize MT5 manager
mt5_manager = MT5Manager()

# ML Agent Integration
class MLAgentClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_analysis(self, symbol: str) -> Optional[MLAnalysis]:
        """Get ML analysis from ML agent"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/ml/analyze/{symbol}",
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return MLAnalysis(**data)
                else:
                    logger.error(f"ML Agent returned status {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting ML analysis: {e}")
            return None
    
    async def get_performance(self) -> Optional[Dict]:
        """Get ML model performance metrics"""
        try:
            async with self.session.get(
                f"{self.base_url}/api/ml/performance",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                return None
        except Exception as e:
            logger.error(f"Error getting ML performance: {e}")
            return None
    
    async def trigger_retrain(self) -> bool:
        """Trigger ML model retraining"""
        try:
            async with self.session.post(
                f"{self.base_url}/api/ml/retrain",
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Error triggering retrain: {e}")
            return False

# Database Manager
class DatabaseManager:
    @staticmethod
    async def initialize_tables():
        """Create database tables if they don't exist"""
        try:
            with SessionLocal() as db:
                # Create signals table
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS signals (
                        id VARCHAR(36) PRIMARY KEY,
                        symbol VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(10, 5) NOT NULL,
                        stop_loss DECIMAL(10, 5) NOT NULL,
                        tp1 DECIMAL(10, 5),
                        tp2 DECIMAL(10, 5),
                        tp3 DECIMAL(10, 5),
                        direction VARCHAR(10) NOT NULL,
                        confidence DECIMAL(5, 4) NOT NULL,
                        pattern VARCHAR(50),
                        risk_reward_ratio DECIMAL(5, 2),
                        market_context JSON,
                        metadata JSON,
                        status VARCHAR(20) DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        executed_at TIMESTAMP,
                        INDEX idx_symbol_date (symbol, created_at),
                        INDEX idx_status (status)
                    )
                """))
                
                # Create trades table
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id VARCHAR(36) PRIMARY KEY,
                        signal_id VARCHAR(36),
                        symbol VARCHAR(10) NOT NULL,
                        direction VARCHAR(10) NOT NULL,
                        entry_price DECIMAL(10, 5) NOT NULL,
                        exit_price DECIMAL(10, 5),
                        stop_loss DECIMAL(10, 5) NOT NULL,
                        take_profit_1 DECIMAL(10, 5),
                        take_profit_2 DECIMAL(10, 5),
                        take_profit_3 DECIMAL(10, 5),
                        position_size DECIMAL(10, 2) NOT NULL,
                        pnl DECIMAL(10, 2),
                        pnl_pips DECIMAL(10, 2),
                        commission DECIMAL(10, 2),
                        swap DECIMAL(10, 2),
                        status VARCHAR(20) DEFAULT 'OPEN',
                        pattern VARCHAR(50),
                        confidence DECIMAL(5, 4),
                        mt5_ticket INTEGER,
                        slippage DECIMAL(10, 5),
                        open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        close_time TIMESTAMP,
                        duration_minutes INTEGER,
                        FOREIGN KEY (signal_id) REFERENCES signals(id),
                        INDEX idx_symbol_status (symbol, status),
                        INDEX idx_open_time (open_time)
                    )
                """))
                
                # Create performance_metrics table
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTO_INCREMENT,
                        date DATE NOT NULL UNIQUE,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate DECIMAL(5, 4) DEFAULT 0,
                        total_pnl DECIMAL(10, 2) DEFAULT 0,
                        gross_profit DECIMAL(10, 2) DEFAULT 0,
                        gross_loss DECIMAL(10, 2) DEFAULT 0,
                        profit_factor DECIMAL(5, 2) DEFAULT 0,
                        average_win DECIMAL(10, 2) DEFAULT 0,
                        average_loss DECIMAL(10, 2) DEFAULT 0,
                        largest_win DECIMAL(10, 2) DEFAULT 0,
                        largest_loss DECIMAL(10, 2) DEFAULT 0,
                        consecutive_wins INTEGER DEFAULT 0,
                        consecutive_losses INTEGER DEFAULT 0,
                        max_drawdown DECIMAL(10, 2) DEFAULT 0,
                        sharpe_ratio DECIMAL(5, 2) DEFAULT 0,
                        sortino_ratio DECIMAL(5, 2) DEFAULT 0,
                        expectancy DECIMAL(10, 2) DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                    )
                """))
                
                # Create system_logs table
                db.execute(text("""
                    CREATE TABLE IF NOT EXISTS system_logs (
                        id INTEGER PRIMARY KEY AUTO_INCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        level VARCHAR(20),
                        component VARCHAR(50),
                        message TEXT,
                        metadata JSON,
                        INDEX idx_timestamp_level (timestamp, level)
                    )
                """))
                
                db.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database tables: {e}")
            # Don't raise exception to allow app to start
    
    @staticmethod
    async def save_signal(signal: TradeSignal) -> str:
        """Save trading signal to database"""
        try:
            with SessionLocal() as db:
                signal_id = str(uuid4())
                
                # Extract take profit levels
                tp_levels = signal.take_profit_levels + [None, None, None]  # Pad with None
                
                query = text("""
                    INSERT INTO signals (id, symbol, entry_price, stop_loss, tp1, tp2, tp3, 
                                       direction, confidence, pattern, risk_reward_ratio, 
                                       market_context, metadata)
                    VALUES (:id, :symbol, :entry_price, :stop_loss, :tp1, :tp2, :tp3,
                           :direction, :confidence, :pattern, :risk_reward_ratio,
                           :market_context, :metadata)
                """)
                
                db.execute(query, {
                    "id": signal_id,
                    "symbol": signal.symbol,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "tp1": tp_levels[0],
                    "tp2": tp_levels[1],
                    "tp3": tp_levels[2],
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "pattern": signal.pattern,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "market_context": json.dumps(signal.market_context),
                    "metadata": json.dumps(signal.metadata) if signal.metadata else None
                })
                db.commit()
                
                # Log to system_logs
                await DatabaseManager.log_event(
                    "INFO", 
                    "SIGNAL", 
                    f"New signal created: {signal.symbol} {signal.direction}",
                    {"signal_id": signal_id, "confidence": signal.confidence}
                )
                
                return signal_id
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            raise HTTPException(status_code=500, detail="Failed to save signal")
    
    @staticmethod
    async def save_trade(signal_id: str, trade_data: Dict) -> str:
        """Save executed trade to database"""
        try:
            with SessionLocal() as db:
                trade_id = str(uuid4())
                
                query = text("""
                    INSERT INTO trades (id, signal_id, symbol, direction, entry_price, stop_loss, 
                                      take_profit_1, position_size, status, pattern, 
                                      confidence, mt5_ticket, slippage, commission)
                    VALUES (:id, :signal_id, :symbol, :direction, :entry_price, :stop_loss,
                           :take_profit_1, :position_size, :status, :pattern,
                           :confidence, :mt5_ticket, :slippage, :commission)
                """)
                
                db.execute(query, {
                    "id": trade_id,
                    "signal_id": signal_id,
                    "symbol": trade_data.get("symbol"),
                    "direction": trade_data.get("direction"),
                    "entry_price": trade_data.get("entry_price"),
                    "stop_loss": trade_data.get("stop_loss"),
                    "take_profit_1": trade_data.get("tp1"),
                    "position_size": trade_data.get("volume", 0.01),
                    "status": "OPEN",
                    "pattern": trade_data.get("pattern"),
                    "confidence": trade_data.get("confidence"),
                    "mt5_ticket": trade_data.get("ticket"),
                    "slippage": trade_data.get("slippage", 0.0),
                    "commission": trade_data.get("commission", 0.0)
                })
                db.commit()
                
                return trade_id
                
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            raise HTTPException(status_code=500, detail="Failed to save trade")
    
    @staticmethod
    async def update_trade_status(trade_id: str, status: str, exit_price: float = None, pnl: float = None):
        """Update trade status and P&L"""
        try:
            with SessionLocal() as db:
                if status == "CLOSED" and exit_price is not None:
                    query = text("""
                        UPDATE trades 
                        SET status = :status, 
                            exit_price = :exit_price, 
                            pnl = :pnl,
                            close_time = CURRENT_TIMESTAMP,
                            duration_minutes = TIMESTAMPDIFF(MINUTE, open_time, CURRENT_TIMESTAMP)
                        WHERE id = :trade_id
                    """)
                    params = {
                        "status": status,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "trade_id": trade_id
                    }
                else:
                    query = text("UPDATE trades SET status = :status WHERE id = :trade_id")
                    params = {"status": status, "trade_id": trade_id}
                
                db.execute(query, params)
                db.commit()
                
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")
    
    @staticmethod
    async def get_performance_metrics(period: str = "all") -> PerformanceMetrics:
        """Get performance metrics for specified period"""
        try:
            with SessionLocal() as db:
                # Build date filter
                date_filter = ""
                if period == "daily":
                    date_filter = "AND DATE(close_time) = CURDATE()"
                elif period == "weekly":
                    date_filter = "AND close_time >= DATE_SUB(CURDATE(), INTERVAL 1 WEEK)"
                elif period == "monthly":
                    date_filter = "AND close_time >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH)"
                
                # Get metrics
                query = text(f"""
                    SELECT 
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                        COALESCE(SUM(pnl), 0) as total_pnl,
                        COALESCE(AVG(pnl), 0) as avg_pnl,
                        COALESCE(MAX(pnl), 0) as largest_win,
                        COALESCE(MIN(pnl), 0) as largest_loss,
                        COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as gross_profit,
                        COALESCE(SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) as gross_loss
                    FROM trades 
                    WHERE status = 'CLOSED' 
                    AND pnl IS NOT NULL
                    {date_filter}
                """)
                
                result = db.execute(query).fetchone()
                
                if result and result.total_trades > 0:
                    # Calculate additional metrics
                    profit_factor = result.gross_profit / result.gross_loss if result.gross_loss > 0 else 0
                    
                    # Get drawdown
                    drawdown_query = text(f"""
                        SELECT MIN(cumulative_pnl) as max_drawdown
                        FROM (
                            SELECT 
                                close_time,
                                SUM(pnl) OVER (ORDER BY close_time) as cumulative_pnl
                            FROM trades
                            WHERE status = 'CLOSED' AND pnl IS NOT NULL
                            {date_filter}
                        ) as cum_pnl
                    """)
                    drawdown_result = db.execute(drawdown_query).fetchone()
                    max_drawdown = abs(drawdown_result.max_drawdown) if drawdown_result and drawdown_result.max_drawdown else 0
                    
                    # Calculate Sharpe ratio (simplified)
                    returns_query = text(f"""
                        SELECT pnl FROM trades 
                        WHERE status = 'CLOSED' AND pnl IS NOT NULL
                        {date_filter}
                        ORDER BY close_time
                    """)
                    returns = [row.pnl for row in db.execute(returns_query)]
                    
                    if len(returns) > 1:
                        returns_array = np.array(returns)
                        sharpe_ratio = np.sqrt(252) * (np.mean(returns_array) / np.std(returns_array)) if np.std(returns_array) > 0 else 0
                        
                        # Sortino ratio (downside deviation)
                        negative_returns = returns_array[returns_array < 0]
                        downside_dev = np.std(negative_returns) if len(negative_returns) > 0 else 0
                        sortino_ratio = np.sqrt(252) * (np.mean(returns_array) / downside_dev) if downside_dev > 0 else 0
                    else:
                        sharpe_ratio = 0
                        sortino_ratio = 0
                    
                    # Get current balance (assuming starting balance of 10000)
                    balance_query = text("SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM trades WHERE status = 'CLOSED'")
                    total_pnl = db.execute(balance_query).fetchone().total_pnl
                    current_balance = 10000 + total_pnl
                    
                    return PerformanceMetrics(
                        total_trades=result.total_trades,
                        win_rate=float(result.win_rate or 0),
                        total_pnl=float(result.total_pnl),
                        avg_pnl=float(result.avg_pnl),
                        max_drawdown=float(max_drawdown),
                        sharpe_ratio=float(sharpe_ratio),
                        sortino_ratio=float(sortino_ratio),
                        profit_factor=float(profit_factor),
                        current_balance=float(current_balance)
                    )
                
                return PerformanceMetrics()
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return PerformanceMetrics()
    
    @staticmethod
    async def get_recent_signals(limit: int = 50) -> List[Dict]:
        """Get recent trading signals"""
        try:
            with SessionLocal() as db:
                query = text("""
                    SELECT 
                        s.*,
                        t.id as trade_id,
                        t.status as trade_status,
                        t.pnl as trade_pnl
                    FROM signals s
                    LEFT JOIN trades t ON s.id = t.signal_id
                    ORDER BY s.created_at DESC
                    LIMIT :limit
                """)
                
                results = db.execute(query, {"limit": limit}).fetchall()
                
                signals = []
                for row in results:
                    signal_dict = dict(row._mapping)
                    # Parse JSON fields
                    if signal_dict.get('market_context'):
                        signal_dict['market_context'] = json.loads(signal_dict['market_context'])
                    if signal_dict.get('metadata'):
                        signal_dict['metadata'] = json.loads(signal_dict['metadata'])
                    signals.append(signal_dict)
                
                return signals
                
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    @staticmethod
    async def get_recent_trades(limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        try:
            with SessionLocal() as db:
                query = text("""
                    SELECT * FROM trades
                    ORDER BY open_time DESC
                    LIMIT :limit
                """)
                
                results = db.execute(query, {"limit": limit}).fetchall()
                return [dict(row._mapping) for row in results]
                
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    @staticmethod
    async def log_event(level: str, component: str, message: str, metadata: Dict = None):
        """Log system event to database"""
        try:
            with SessionLocal() as db:
                query = text("""
                    INSERT INTO system_logs (level, component, message, metadata)
                    VALUES (:level, :component, :message, :metadata)
                """)
                
                db.execute(query, {
                    "level": level,
                    "component": component,
                    "message": message,
                    "metadata": json.dumps(metadata) if metadata else None
                })
                db.commit()
                
        except Exception as e:
            logger.error(f"Error logging event: {e}")

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
        
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        message_json = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to websocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

manager = ConnectionManager()

# Startup and shutdown events
async def startup_event():
    global redis_client
    
    logger.info("Starting ML Trading Backend...")
    
    # Initialize database tables
    await DatabaseManager.initialize_tables()
    
    # Initialize Redis
    try:
        redis_client = await redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=10,
            socket_timeout=10,
            retry_on_timeout=True
        )
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        redis_client = None
    
    # Initialize MT5 if enabled
    if system_config.mt5_enabled:
        mt5_account = os.getenv("MT5_ACCOUNT")
        mt5_password = os.getenv("MT5_PASSWORD")
        mt5_server = os.getenv("MT5_SERVER")
        
        if mt5_account and mt5_password:
            success = await mt5_manager.connect(int(mt5_account), mt5_password, mt5_server)
            if not success:
                logger.warning("MT5 connection failed, will retry later")
        else:
            logger.info("MT5 credentials not provided, using demo mode")
    
    # Load system configuration from cache
    if redis_client:
        try:
            config_json = await redis_client.get("system:config")
            if config_json:
                config_dict = json.loads(config_json)
                global system_config
                system_config = SystemConfig(**config_dict)
                logger.info("Loaded system configuration from cache")
        except Exception as e:
            logger.error(f"Error loading config from cache: {e}")
    
    logger.info("ML Trading Backend started successfully")

async def shutdown_event():
    logger.info("Shutting down ML Trading Backend...")
    
    # Close Redis connection
    if redis_client:
        await redis_client.close()
    
    # Disconnect MT5
    mt5_manager.disconnect()
    
    # Close all WebSocket connections
    for ws in manager.active_connections.copy():
        await ws.close()
    
    logger.info("ML Trading Backend shut down")

# API Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ML Trading System Backend",
        "version": "3.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "api_docs": "/docs",
            "websocket": "/ws",
            "ml_analysis": "/api/ml/analyze/{symbol}",
            "signals": "/api/signals",
            "trades": "/api/trades",
            "positions": "/api/positions",
            "performance": "/api/performance"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": "unknown",
            "redis": "disconnected",
            "mt5": "disconnected",
            "ml_agent": "unknown"
        }
    }
    
    # Check database
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        health_status["components"]["database"] = "connected"
    except Exception as e:
        health_status["components"]["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    if redis_client:
        try:
            await redis_client.ping()
            health_status["components"]["redis"] = "connected"
        except:
            health_status["components"]["redis"] = "error"
            health_status["status"] = "degraded"
    
    # Check MT5
    health_status["components"]["mt5"] = "connected" if mt5_manager.connected else "disconnected"
    
    # Check ML Agent
    try:
        async with MLAgentClient(ML_AGENT_URL) as client:
            perf = await client.get_performance()
            if perf:
                health_status["components"]["ml_agent"] = "connected"
            else:
                health_status["components"]["ml_agent"] = "error"
                health_status["status"] = "degraded"
    except:
        health_status["components"]["ml_agent"] = "disconnected"
        health_status["status"] = "degraded"
    
    return health_status

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "data": {
                "status": "connected",
                "timestamp": datetime.now().isoformat(),
                "mt5_connected": mt5_manager.connected
            }
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages with timeout
                message = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                # Parse and handle message
                try:
                    data = json.loads(message)
                    
                    # Handle different message types
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                    elif data.get("type") == "subscribe":
                        # Handle subscription requests
                        symbol = data.get("symbol")
                        if symbol:
                            await websocket.send_json({
                                "type": "subscribed",
                                "symbol": symbol,
                                "timestamp": datetime.now().isoformat()
                            })
                            
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ML Integration endpoints
@app.get("/api/ml/analyze/{symbol}")
async def get_ml_analysis(symbol: str):
    """Get ML analysis for a symbol"""
    try:
        # First check cache
        if redis_client:
            cached = await redis_client.get(f"ml_analysis:{symbol}")
            if cached:
                return json.loads(cached)
        
        # Get from ML agent
        async with MLAgentClient(ML_AGENT_URL) as client:
            analysis = await client.get_analysis(symbol)
            
            if analysis:
                # Cache for 5 minutes
                if redis_client:
                    await redis_client.setex(
                        f"ml_analysis:{symbol}",
                        300,
                        json.dumps(analysis.dict(), default=str)
                    )
                
                # Broadcast to WebSocket clients
                await manager.broadcast({
                    "type": "ml_analysis_update",
                    "data": analysis.dict()
                })
                
                return analysis
            else:
                raise HTTPException(status_code=503, detail="ML Agent not available")
                
    except Exception as e:
        logger.error(f"Error getting ML analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get ML analysis")

@app.get("/api/ml/performance")
async def get_ml_performance():
    """Get ML model performance metrics"""
    try:
        async with MLAgentClient(ML_AGENT_URL) as client:
            performance = await client.get_performance()
            
            if performance:
                return performance
            else:
                return {
                    "ml_model": {"is_trained": False},
                    "signals": {"total_generated": 0, "last_24h": 0},
                    "system": {"status": "ml_agent_unavailable"}
                }
                
    except Exception as e:
        logger.error(f"Error getting ML performance: {e}")
        return {
            "ml_model": {"is_trained": False},
            "signals": {"total_generated": 0, "last_24h": 0},
            "system": {"status": "error", "message": str(e)}
        }

@app.post("/api/ml/retrain")
async def trigger_ml_retrain(background_tasks: BackgroundTasks):
    """Trigger ML model retraining"""
    try:
        async with MLAgentClient(ML_AGENT_URL) as client:
            success = await client.trigger_retrain()
            
            if success:
                # Broadcast retrain event
                await manager.broadcast({
                    "type": "ml_retrain_started",
                    "timestamp": datetime.now().isoformat()
                })
                
                return {"status": "success", "message": "Model retraining initiated"}
            else:
                raise HTTPException(status_code=503, detail="Failed to trigger retraining")
                
    except Exception as e:
        logger.error(f"Error triggering ML retrain: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

# Trading endpoints
@app.post("/api/signals", response_model=Dict)
async def create_signal(signal: TradeSignal, background_tasks: BackgroundTasks):
    """Create a new trading signal"""
    try:
        # Validate signal
        if signal.confidence < system_config.min_confidence:
            raise HTTPException(
                status_code=400,
                detail=f"Signal confidence {signal.confidence} below minimum {system_config.min_confidence}"
            )
        
        if signal.symbol not in system_config.allowed_symbols:
            raise HTTPException(
                status_code=400,
                detail=f"Symbol {signal.symbol} not in allowed list"
            )
        
        # Save signal to database
        signal_id = await DatabaseManager.save_signal(signal)
        
        # Cache in Redis
        if redis_client:
            await redis_client.setex(
                f"signal:{signal_id}",
                3600,  # 1 hour expiry
                json.dumps(signal.dict(), default=str)
            )
            
            # Increment daily signal counter
            today = datetime.now().strftime("%Y%m%d")
            await redis_client.incr(f"signals_count:{today}")
            await redis_client.expire(f"signals_count:{today}", 86400)
        
        # Check if auto-trading is enabled
        if system_config.enable_auto_trading and mt5_manager.connected:
            # Check daily trade limit
            if redis_client:
                daily_count = await redis_client.get(f"trades_count:{today}")
                if daily_count and int(daily_count) >= system_config.max_daily_trades:
                    logger.warning(f"Daily trade limit reached: {daily_count}/{system_config.max_daily_trades}")
                else:
                    # Execute trade in background
                    background_tasks.add_task(execute_signal_trade, signal_id, signal)
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "new_signal",
            "data": {
                "id": signal_id,
                **signal.dict(exclude_unset=True),
                "timestamp": datetime.now().isoformat()
            }
        })
        
        # Log to database
        await DatabaseManager.log_event(
            "INFO",
            "SIGNALS",
            f"New signal created: {signal.symbol} {signal.direction}",
            {"signal_id": signal_id, "confidence": signal.confidence}
        )
        
        return {
            "status": "success",
            "signal_id": signal_id,
            "message": "Signal created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating signal: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to create signal")

async def execute_signal_trade(signal_id: str, signal: TradeSignal):
    """Execute trade for a signal (background task)"""
    try:
        # Check current positions
        positions = await mt5_manager.get_positions()
        open_positions = len([p for p in positions if p['symbol'] == signal.symbol])
        
        if open_positions >= system_config.max_concurrent_trades:
            logger.warning(f"Max concurrent trades reached for {signal.symbol}")
            return
        
        # Calculate position size
        position_size = 0.01  # Default, will be calculated based on risk
        
        # Execute trade on MT5
        trade_result = await mt5_manager.execute_trade(signal, position_size)
        
        if trade_result.get("status") == "success":
            # Save trade to database
            trade_data = {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": trade_result.get("price", signal.entry_price),
                "stop_loss": signal.stop_loss,
                "tp1": signal.take_profit_levels[0] if signal.take_profit_levels else None,
                "volume": trade_result.get("volume", position_size),
                "pattern": signal.pattern,
                "confidence": signal.confidence,
                "ticket": trade_result.get("ticket"),
                "slippage": trade_result.get("slippage", 0),
                "commission": trade_result.get("commission", 0)
            }
            
            trade_id = await DatabaseManager.save_trade(signal_id, trade_data)
            
            # Update signal status
            with SessionLocal() as db:
                query = text("UPDATE signals SET status = :status, executed_at = CURRENT_TIMESTAMP WHERE id = :id")
                db.execute(query, {"status": "executed", "id": signal_id})
                db.commit()
            
            # Update daily trade counter
            if redis_client:
                today = datetime.now().strftime("%Y%m%d")
                await redis_client.incr(f"trades_count:{today}")
            
            # Broadcast trade execution
            await manager.broadcast({
                "type": "trade_executed",
                "data": {
                    "trade_id": trade_id,
                    "signal_id": signal_id,
                    "ticket": trade_result.get("ticket"),
                    "price": trade_result.get("price"),
                    "volume": trade_result.get("volume"),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            # Log event
            await DatabaseManager.log_event(
                "INFO",
                "TRADING",
                f"Trade executed: {signal.symbol} {signal.direction}",
                {
                    "trade_id": trade_id,
                    "signal_id": signal_id,
                    "ticket": trade_result.get("ticket")
                }
            )
            
        else:
            # Update signal status to rejected
            with SessionLocal() as db:
                query = text("UPDATE signals SET status = :status WHERE id = :id")
                db.execute(query, {"status": "rejected", "id": signal_id})
                db.commit()
            
            logger.error(f"Trade execution failed: {trade_result}")
            
            # Log failure
            await DatabaseManager.log_event(
                "ERROR",
                "TRADING",
                f"Trade execution failed: {trade_result.get('message', 'Unknown error')}",
                {"signal_id": signal_id, "error": trade_result}
            )
            
    except Exception as e:
        logger.error(f"Error executing signal trade: {e}")
        logger.error(traceback.format_exc())
        
        # Log error
        await DatabaseManager.log_event(
            "ERROR",
            "TRADING",
            f"Trade execution error: {str(e)}",
            {"signal_id": signal_id}
        )

@app.get("/api/signals")
async def get_signals(limit: int = 50, status: Optional[str] = None):
    """Get recent trading signals"""
    try:
        signals = await DatabaseManager.get_recent_signals(limit)
        
        # Filter by status if provided
        if status:
            signals = [s for s in signals if s.get('status') == status]
        
        return {
            "signals": signals,
            "count": len(signals),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        raise HTTPException(status_code=500, detail="Failed to get signals")

@app.get("/api/trades")
async def get_trades(limit: int = 100, status: Optional[str] = None):
    """Get recent trades"""
    try:
        trades = await DatabaseManager.get_recent_trades(limit)
        
        # Filter by status if provided
        if status:
            trades = [t for t in trades if t.get('status') == status]
        
        # Calculate current P&L for open trades
        if mt5_manager.connected:
            positions = await mt5_manager.get_positions()
            position_map = {p['ticket']: p for p in positions}
            
            for trade in trades:
                if trade['status'] == 'OPEN' and trade.get('mt5_ticket') in position_map:
                    position = position_map[trade['mt5_ticket']]
                    trade['current_price'] = position['price_current']
                    trade['current_pnl'] = position['profit']
        
        return {
            "trades": trades,
            "count": len(trades),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trades")

@app.get("/api/positions")
async def get_positions():
    """Get current MT5 positions"""
    try:
        positions = await mt5_manager.get_positions()
        
        return {
            "positions": positions,
            "count": len(positions),
            "total_pnl": sum(p.get('profit', 0) for p in positions),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")

@app.post("/api/positions/{ticket}/close")
async def close_position(ticket: int):
    """Close a specific position"""
    try:
        result = await mt5_manager.close_position(ticket)
        
        if result.get("status") == "success":
            # Update trade in database
            with SessionLocal() as db:
                query = text("""
                    UPDATE trades 
                    SET status = 'CLOSED',
                        exit_price = :exit_price,
                        pnl = :pnl,
                        close_time = CURRENT_TIMESTAMP
                    WHERE mt5_ticket = :ticket
                """)
                
                db.execute(query, {
                    "exit_price": result.get("price"),
                    "pnl": result.get("profit"),
                    "ticket": ticket
                })
                db.commit()
            
            # Broadcast position closed
            await manager.broadcast({
                "type": "position_closed",
                "data": {
                    "ticket": ticket,
                    "price": result.get("price"),
                    "profit": result.get("profit"),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
            return {"status": "success", "message": "Position closed successfully", "result": result}
        else:
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to close position"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail="Failed to close position")

@app.get("/api/performance")
async def get_performance(period: Optional[str] = "all"):
    """Get trading performance metrics"""
    try:
        # Validate period
        valid_periods = ["all", "daily", "weekly", "monthly"]
        if period not in valid_periods:
            period = "all"
        
        # Get metrics from database
        metrics = await DatabaseManager.get_performance_metrics(period)
        
        # Add additional metrics if MT5 is connected
        if mt5_manager.connected and mt5_manager.account_info:
            metrics.current_balance = mt5_manager.account_info.get('balance', metrics.current_balance)
        
        # Get performance by symbol
        symbol_performance = {}
        if period == "all":
            with SessionLocal() as db:
                query = text("""
                    SELECT 
                        symbol,
                        COUNT(*) as trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl
                    FROM trades
                    WHERE status = 'CLOSED' AND pnl IS NOT NULL
                    GROUP BY symbol
                """)
                
                results = db.execute(query).fetchall()
                for row in results:
                    symbol_performance[row.symbol] = {
                        "trades": row.trades,
                        "win_rate": row.wins / row.trades if row.trades > 0 else 0,
                        "total_pnl": float(row.total_pnl or 0),
                        "avg_pnl": float(row.avg_pnl or 0)
                    }
        
        return {
            "period": period,
            "metrics": metrics.dict(),
            "symbol_performance": symbol_performance,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get performance metrics")

@app.get("/api/market-data/{symbol}")
async def get_market_data(
    symbol: str = "USDJPY", 
    timeframe: str = "M15", 
    count: int = 1000
):
    """Get market data from MT5 or mock data"""
    try:
        # Validate inputs
        if symbol not in system_config.allowed_symbols:
            raise HTTPException(status_code=400, detail=f"Symbol {symbol} not allowed")
        
        valid_timeframes = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
        if timeframe not in valid_timeframes:
            timeframe = "M15"
        
        # Get market data
        data = await mt5_manager.get_market_data(symbol, timeframe, count)
        
        if data:
            # Cache in Redis
            if redis_client:
                await redis_client.setex(
                    f"market_data:{symbol}:{timeframe}",
                    60,  # 1 minute cache
                    json.dumps(data, default=str)
                )
            
            return data
        else:
            raise HTTPException(status_code=404, detail="No market data available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market data")

@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    return system_config.dict()

@app.put("/api/config")
async def update_config(config: SystemConfig):
    """Update system configuration"""
    try:
        global system_config
        system_config = config
        
        # Save to Redis
        if redis_client:
            await redis_client.set(
                "system:config",
                json.dumps(config.dict())
            )
        
        # Broadcast config update
        await manager.broadcast({
            "type": "config_updated",
            "data": config.dict(),
            "timestamp": datetime.now().isoformat()
        })
        
        # Log event
        await DatabaseManager.log_event(
            "INFO",
            "SYSTEM",
            "Configuration updated",
            config.dict()
        )
        
        return {"status": "success", "message": "Configuration updated"}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@app.get("/api/reports/{report_type}")
async def generate_report(
    report_type: str,
    period: str = "monthly",
    format: str = "json"
):
    """Generate trading reports"""
    try:
        valid_types = ["performance", "trades", "signals", "risk"]
        if report_type not in valid_types:
            raise HTTPException(status_code=400, detail=f"Invalid report type: {report_type}")
        
        report_data = {}
        
        if report_type == "performance":
            # Performance report
            report_data["metrics"] = (await DatabaseManager.get_performance_metrics(period)).dict()
            report_data["trades"] = await DatabaseManager.get_recent_trades(1000)
            
        elif report_type == "trades":
            # Detailed trades report
            trades = await DatabaseManager.get_recent_trades(1000)
            report_data["trades"] = trades
            report_data["summary"] = {
                "total": len(trades),
                "open": len([t for t in trades if t['status'] == 'OPEN']),
                "closed": len([t for t in trades if t['status'] == 'CLOSED']),
                "profitable": len([t for t in trades if t.get('pnl', 0) > 0]),
                "loss": len([t for t in trades if t.get('pnl', 0) < 0])
            }
            
        elif report_type == "signals":
            # Signals analysis report
            signals = await DatabaseManager.get_recent_signals(1000)
            report_data["signals"] = signals
            report_data["summary"] = {
                "total": len(signals),
                "executed": len([s for s in signals if s['status'] == 'executed']),
                "pending": len([s for s in signals if s['status'] == 'pending']),
                "rejected": len([s for s in signals if s['status'] == 'rejected']),
                "avg_confidence": np.mean([s['confidence'] for s in signals]) if signals else 0
            }
            
        elif report_type == "risk":
            # Risk analysis report
            metrics = await DatabaseManager.get_performance_metrics()
            positions = await mt5_manager.get_positions()
            
            report_data["current_exposure"] = {
                "open_positions": len(positions),
                "total_risk": sum(abs(p.get('profit', 0)) for p in positions),
                "symbols": list(set(p['symbol'] for p in positions))
            }
            report_data["risk_metrics"] = {
                "max_drawdown": metrics.max_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "win_rate": metrics.win_rate
            }
        
        # Add metadata
        report_data["metadata"] = {
            "report_type": report_type,
            "period": period,
            "generated_at": datetime.now().isoformat(),
            "format": format
        }
        
        # Log report generation
        await DatabaseManager.log_event(
            "INFO",
            "REPORTS",
            f"Generated {report_type} report",
            {"period": period, "format": format}
        )
        
        if format == "json":
            return report_data
        else:
            # Could implement CSV, PDF, etc.
            raise HTTPException(status_code=501, detail=f"Format {format} not implemented")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate report")

@app.get("/api/logs")
async def get_system_logs(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 100
):
    """Get system logs"""
    try:
        with SessionLocal() as db:
            query = "SELECT * FROM system_logs WHERE 1=1"
            params = {}
            
            if level:
                query += " AND level = :level"
                params["level"] = level
            
            if component:
                query += " AND component = :component"
                params["component"] = component
            
            query += " ORDER BY timestamp DESC LIMIT :limit"
            params["limit"] = limit
            
            results = db.execute(text(query), params).fetchall()
            
            logs = []
            for row in results:
                log_dict = dict(row._mapping)
                if log_dict.get('metadata'):
                    log_dict['metadata'] = json.loads(log_dict['metadata'])
                logs.append(log_dict)
            
            return {
                "logs": logs,
                "count": len(logs),
                "filters": {"level": level, "component": component}
            }
            
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get logs")

# Monitoring endpoint for Railway
@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "connections": {
                "websocket_clients": len(manager.active_connections),
                "mt5_connected": mt5_manager.connected,
                "redis_connected": redis_client is not None
            },
            "trading": {
                "auto_trading_enabled": system_config.enable_auto_trading,
                "open_positions": len(await mt5_manager.get_positions()) if mt5_manager.connected else 0
            }
        }
        
        # Add daily counts from Redis
        if redis_client:
            today = datetime.now().strftime("%Y%m%d")
            metrics["daily_counts"] = {
                "signals": int(await redis_client.get(f"signals_count:{today}") or 0),
                "trades": int(await redis_client.get(f"trades_count:{today}") or 0)
            }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": str(e)}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    await DatabaseManager.log_event(
        "ERROR",
        "HTTP",
        f"{exc.status_code}: {exc.detail}",
        {"path": str(request.url), "method": request.method}
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    
    await DatabaseManager.log_event(
        "ERROR",
        "SYSTEM",
        f"Unhandled exception: {str(exc)}",
        {"path": str(request.url), "method": request.method, "traceback": traceback.format_exc()}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # Run with uvicorn
    port = int(os.getenv("PORT", 8080))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level="info"
    )
