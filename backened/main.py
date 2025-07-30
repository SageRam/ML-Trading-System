# main.py - Enhanced FastAPI Backend with MT5 Integration
import asyncio
import json
import logging
import os
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import MetaTrader5 as mt5

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://trading_user:trading_password@localhost:5432/trading_system")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Pydantic models
class TradeSignal(BaseModel):
    symbol: str = "USDJPY"
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: Optional[float] = None
    tp3: Optional[float] = None
    direction: str  # BUY or SELL
    confidence: float
    pattern: str
    risk_reward_ratio: float
    expected_value: float
    market_context: Optional[Dict] = None
    news_impact: Optional[float] = 0.0

class TradeExecution(BaseModel):
    signal_id: str
    mt5_ticket: Optional[int] = None
    execution_price: float
    slippage: Optional[float] = 0.0
    commission: Optional[float] = 0.0
    status: str

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

class SystemConfig(BaseModel):
    max_concurrent_trades: int = 3
    enable_auto_trading: bool = False
    risk_per_trade: float = 2.0
    storage_location: str = "local"  # local or cloud
    mt5_enabled: bool = True

# FastAPI app
app = FastAPI(
    title="Advanced Trading System API",
    description="Comprehensive trading system with MT5 integration and ML analytics",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
redis_client = None
active_connections: List[WebSocket] = []
logger = logging.getLogger(__name__)

# MT5 Integration Class
class MT5Manager:
    def __init__(self):
        self.connected = False
        self.account_info = {}
        
    async def connect(self, account: int = None, password: str = None, server: str = None):
        """Connect to MT5"""
        try:
            if not mt5.initialize():
                logger.error("Failed to initialize MT5")
                return False
                
            if account and password:
                if not mt5.login(account, password, server):
                    logger.error(f"Failed to login to account {account}")
                    return False
                    
            self.connected = True
            self.account_info = mt5.account_info()._asdict() if mt5.account_info() else {}
            logger.info(f"MT5 connected successfully. Account: {self.account_info.get('login', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")
    
    async def get_market_data(self, symbol: str = "USDJPY", timeframe: str = "M15", count: int = 1000):
        """Get market data from MT5"""
        if not self.connected:
            raise HTTPException(status_code=500, detail="MT5 not connected")
            
        try:
            # Convert timeframe string to MT5 constant
            tf_map = {
                "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }
            
            timeframe_mt5 = tf_map.get(timeframe, mt5.TIMEFRAME_M15)
            rates = mt5.copy_rates_from_pos(symbol, timeframe_mt5, 0, count)
            
            if rates is None:
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': df.to_dict('records'),
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def execute_trade(self, signal: TradeSignal) -> Dict:
        """Execute trade on MT5"""
        if not self.connected:
            raise HTTPException(status_code=500, detail="MT5 not connected")
            
        try:
            symbol = signal.symbol
            lot = 0.01  # Default lot size
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return {"status": "error", "message": f"Symbol {symbol} not found"}
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": mt5.ORDER_TYPE_BUY if signal.direction == "BUY" else mt5.ORDER_TYPE_SELL,
                "price": signal.entry_price,
                "sl": signal.stop_loss,
                "tp": signal.tp1,
                "comment": f"ML Signal - {signal.pattern}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return {
                    "status": "error",
                    "message": f"Order failed: {result.comment}",
                    "retcode": result.retcode
                }
            
            return {
                "status": "success",
                "ticket": result.order,
                "price": result.price,
                "volume": result.volume,
                "comment": result.comment
            }
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_positions(self) -> List[Dict]:
        """Get open positions"""
        if not self.connected:
            return []
            
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
                
            return [
                {
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "time": datetime.fromtimestamp(pos.time).isoformat()
                }
                for pos in positions
            ]
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

# Initialize MT5 manager
mt5_manager = MT5Manager()

# Database operations
class DatabaseManager:
    @staticmethod
    async def save_signal(signal: TradeSignal) -> str:
        """Save trading signal to database"""
        try:
            with SessionLocal() as db:
                signal_id = str(uuid4())
                query = text("""
                    INSERT INTO signals (id, symbol, entry_price, stop_loss, tp1, tp2, tp3, 
                                       direction, confidence, pattern, risk_reward_ratio, 
                                       expected_value, market_context, news_impact)
                    VALUES (:id, :symbol, :entry_price, :stop_loss, :tp1, :tp2, :tp3,
                           :direction, :confidence, :pattern, :risk_reward_ratio,
                           :expected_value, :market_context, :news_impact)
                """)
                
                db.execute(query, {
                    "id": signal_id,
                    "symbol": signal.symbol,
                    "entry_price": signal.entry_price,
                    "stop_loss": signal.stop_loss,
                    "tp1": signal.tp1,
                    "tp2": signal.tp2,
                    "tp3": signal.tp3,
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "pattern": signal.pattern,
                    "risk_reward_ratio": signal.risk_reward_ratio,
                    "expected_value": signal.expected_value,
                    "market_context": json.dumps(signal.market_context or {}),
                    "news_impact": signal.news_impact
                })
                db.commit()
                return signal_id
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            raise HTTPException(status_code=500, detail="Failed to save signal")
    
    @staticmethod
    async def save_trade(trade_data: Dict) -> str:
        """Save executed trade to database"""
        try:
            with SessionLocal() as db:
                trade_id = str(uuid4())
                query = text("""
                    INSERT INTO trades (id, symbol, direction, entry_price, stop_loss, 
                                      take_profit_1, position_size, status, pattern, 
                                      confidence, mt5_ticket, slippage)
                    VALUES (:id, :symbol, :direction, :entry_price, :stop_loss,
                           :take_profit_1, :position_size, :status, :pattern,
                           :confidence, :mt5_ticket, :slippage)
                """)
                
                db.execute(query, {
                    "id": trade_id,
                    "symbol": trade_data.get("symbol", "USDJPY"),
                    "direction": trade_data.get("direction"),
                    "entry_price": trade_data.get("entry_price"),
                    "stop_loss": trade_data.get("stop_loss"),
                    "take_profit_1": trade_data.get("tp1"),
                    "position_size": trade_data.get("volume", 0.01),
                    "status": "OPEN",
                    "pattern": trade_data.get("pattern"),
                    "confidence": trade_data.get("confidence"),
                    "mt5_ticket": trade_data.get("ticket"),
                    "slippage": trade_data.get("slippage", 0.0)
                })
                db.commit()
                return trade_id
                
        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            raise HTTPException(status_code=500, detail="Failed to save trade")
    
    @staticmethod
    async def get_performance_metrics() -> PerformanceMetrics:
        """Get current performance metrics"""
        try:
            with SessionLocal() as db:
                # Get trades data
                query = text("""
                    SELECT COUNT(*) as total_trades,
                           AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                           SUM(pnl) as total_pnl,
                           AVG(pnl) as avg_pnl
                    FROM trades 
                    WHERE status = 'CLOSED' AND pnl IS NOT NULL
                """)
                
                result = db.execute(query).fetchone()
                
                if result:
                    return PerformanceMetrics(
                        total_trades=result.total_trades or 0,
                        win_rate=float(result.win_rate or 0),
                        total_pnl=float(result.total_pnl or 0),
                        avg_pnl=float(result.avg_pnl or 0)
                    )
                else:
                    return PerformanceMetrics()
                    
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return PerformanceMetrics()

# WebSocket manager
async def broadcast_message(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    if active_connections:
        message_str = json.dumps(message)
        for connection in active_connections.copy():
            try:
                await connection.send_text(message_str)
            except WebSocketDisconnect:
                active_connections.remove(connection)

# Startup event
@app.on_event("startup")
async def startup_event():
    global redis_client
    
    # Initialize Redis
    try:
        redis_client = redis.from_url(REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    # Initialize MT5
    if os.getenv("MT5_ENABLED", "true").lower() == "true":
        account = os.getenv("MT5_ACCOUNT")
        password = os.getenv("MT5_PASSWORD")
        server = os.getenv("MT5_SERVER")
        
        if account and password:
            await mt5_manager.connect(int(account), password, server)
        else:
            logger.warning("MT5 credentials not provided, connecting without login")
            await mt5_manager.connect()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    if redis_client:
        await redis_client.close()
    
    mt5_manager.disconnect()

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mt5_connected": mt5_manager.connected,
        "redis_connected": redis_client is not None
    }

@app.post("/api/signals")
async def create_signal(signal: TradeSignal, background_tasks: BackgroundTasks):
    """Create a new trading signal"""
    try:
        # Save signal to database
        signal_id = await DatabaseManager.save_signal(signal)
        
        # Cache in Redis
        if redis_client:
            await redis_client.setex(
                f"signal:{signal_id}",
                3600,  # 1 hour expiry
                json.dumps(signal.dict())
            )
        
        # Check if auto-trading is enabled
        config = await get_system_config()
        if config.enable_auto_trading and mt5_manager.connected:
            # Execute trade in background
            background_tasks.add_task(execute_signal_trade, signal_id, signal)
        
        # Broadcast to WebSocket clients
        await broadcast_message({
            "type": "new_signal",
            "data": {
                "id": signal_id,
                **signal.dict(),
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {"status": "success", "signal_id": signal_id}
        
    except Exception as e:
        logger.error(f"Error creating signal: {e}")
        raise HTTPException(status_code=500, detail="Failed to create signal")

async def execute_signal_trade(signal_id: str, signal: TradeSignal):
    """Execute trade for a signal (background task)"""
    try:
        # Execute trade on MT5
        trade_result = await mt5_manager.execute_trade(signal)
        
        if trade_result.get("status") == "success":
            # Save trade to database
            trade_data = {
                "symbol": signal.symbol,
                "direction": signal.direction,
                "entry_price": trade_result.get("price", signal.entry_price),
                "stop_loss": signal.stop_loss,
                "tp1": signal.tp1,
                "volume": trade_result.get("volume", 0.01),
                "pattern": signal.pattern,
                "confidence": signal.confidence,
                "ticket": trade_result.get("ticket"),
                "slippage": abs(trade_result.get("price", signal.entry_price) - signal.entry_price)
            }
            
            trade_id = await DatabaseManager.save_trade(trade_data)
            
            # Update signal status
            with SessionLocal() as db:
                query = text("UPDATE signals SET status = :status, executed_at = NOW() WHERE id = :id")
                db.execute(query, {"status": "executed", "id": signal_id})
                db.commit()
            
            # Broadcast trade execution
            await broadcast_message({
                "type": "trade_executed",
                "data": {
                    "trade_id": trade_id,
                    "signal_id": signal_id,
                    "ticket": trade_result.get("ticket"),
                    "price": trade_result.get("price"),
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        else:
            logger.error(f"Trade execution failed: {trade_result}")
            
    except Exception as e:
        logger.error(f"Error executing signal trade: {e}")

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str = "USDJPY", timeframe: str = "M15", count: int = 1000):
    """Get market data from MT5"""
    try:
        data = await mt5_manager.get_market_data(symbol, timeframe, count)
        if data is None:
            raise HTTPException(status_code=404, detail="No market data available")
        return data
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail="Failed to get market data")

@app.get("/api/positions")
async def get_positions():
    """Get current MT5 positions"""
    try:
        positions = await mt5_manager.get_positions()
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")

@app.get("/api/performance")
async def get_performance():