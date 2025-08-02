# ML Trading System Backend

Production-ready FastAPI backend for the ML Trading System with MetaTrader 5 integration and real-time WebSocket support.

## 🚀 Features

### Core Functionality
- **ML Integration**: Seamless connection with ML Agent for predictions
- **MT5 Trading**: Automated trade execution via MetaTrader 5
- **Real-time Updates**: WebSocket support for live data streaming
- **Risk Management**: Position sizing and daily trade limits
- **Performance Tracking**: Comprehensive metrics and reporting

### Technical Features
- **FastAPI Framework**: High-performance async API
- **PostgreSQL + Redis**: Dual database system for persistence and caching
- **WebSocket Support**: Real-time bidirectional communication
- **Health Monitoring**: Built-in health checks and metrics
- **CORS Support**: Configured for Vercel frontend

## 📋 Prerequisites

- Python 3.9+
- PostgreSQL (Railway)
- Redis (Railway)
- MetaTrader 5 (optional, for live trading)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/SageRam/ML-Trading-System.git
cd ML-Trading-System/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

## 🚀 Deployment

### Railway Deployment

1. **Create Railway Project**
   ```bash
   railway login
   railway init
   ```

2. **Add Services**
   - Add PostgreSQL service
   - Add Redis service
   - Note connection strings

3. **Configure Environment Variables**
   ```bash
   railway variables set DATABASE_URL="postgresql://..."
   railway variables set REDIS_URL="redis://..."
   railway variables set ML_AGENT_URL="https://your-ml-agent.railway.app"
   ```

4. **Deploy**
   ```bash
   railway up
   ```

### Docker Deployment

```bash
# Build image
docker build -t ml-trading-backend .

# Run container
docker run -p 8080:8080 --env-file .env ml-trading-backend
```

## 📡 API Endpoints

### Health & Status
- `GET /` - API info and available endpoints
- `GET /health` - Comprehensive health check
- `GET /api/metrics` - System metrics

### ML Integration
- `GET /api/ml/analyze/{symbol}` - Get ML analysis
- `GET /api/ml/performance` - ML model performance
- `POST /api/ml/retrain` - Trigger model retraining

### Trading
- `POST /api/signals` - Create trading signal
- `GET /api/signals` - Get recent signals
- `GET /api/trades` - Get trades
- `GET /api/positions` - Get open positions
- `POST /api/positions/{ticket}/close` - Close position

### Market Data
- `GET /api/market-data/{symbol}` - Get market data
- `WS /ws` - WebSocket connection

### Performance & Reports
- `GET /api/performance` - Performance metrics
- `GET /api/reports/{type}` - Generate reports

### Configuration
- `GET /api/config` - Get system config
- `PUT /api/config` - Update config

## 🔌 WebSocket Events

### Client → Server
```json
{
  "type": "subscribe",
  "symbol": "USDJPY"
}
```

### Server → Client
```json
{
  "type": "new_signal",
  "data": {
    "id": "signal-id",
    "symbol": "USDJPY",
    "direction": "BUY",
    "confidence": 0.85
  }
}
```

### Event Types
- `connection` - Initial connection
- `new_signal` - Trading signal generated
- `trade_executed` - Trade executed
- `position_closed` - Position closed
- `ml_analysis_update` - ML analysis update
- `config_updated` - Configuration changed

## 🗄️ Database Schema

### Signals Table
```sql
CREATE TABLE signals (
    id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    entry_price DECIMAL(10, 5),
    stop_loss DECIMAL(10, 5),
    tp1 DECIMAL(10, 5),
    direction VARCHAR(10),
    confidence DECIMAL(5, 4),
    pattern VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Trades Table
```sql
CREATE TABLE trades (
    id VARCHAR(36) PRIMARY KEY,
    signal_id VARCHAR(36),
    symbol VARCHAR(10) NOT NULL,
    direction VARCHAR(10),
    entry_price DECIMAL(10, 5),
    exit_price DECIMAL(10, 5),
    pnl DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'OPEN',
    open_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    close_time TIMESTAMP
);
```

## 🔧 Configuration

### System Configuration
```python
{
    "max_concurrent_trades": 3,
    "enable_auto_trading": false,
    "risk_per_trade": 2.0,
    "mt5_enabled": true,
    "allowed_symbols": ["USDJPY", "EURUSD", "GBPUSD"],
    "min_confidence": 0.7,
    "max_daily_trades": 10
}
```

### Environment Variables
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `ML_AGENT_URL` - ML Agent service URL
- `VERCEL_FRONTEND_URL` - Frontend URL for CORS
- `MT5_ACCOUNT` - MetaTrader 5 account
- `MT5_PASSWORD` - MetaTrader 5 password
- `MT5_SERVER` - MetaTrader 5 server

## 📊 Monitoring

### Health Check Response
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00Z",
  "components": {
    "database": "connected",
    "redis": "connected",
    "mt5": "connected",
    "ml_agent": "connected"
  }
}
```

### Metrics Endpoint
```json
{
  "connections": {
    "websocket_clients": 5,
    "mt5_connected": true
  },
  "trading": {
    "auto_trading_enabled": false,
    "open_positions": 2
  },
  "daily_counts": {
    "signals": 45,
    "trades": 12
  }
}
```

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Verify DATABASE_URL is correct
   - Check PostgreSQL SSL requirements
   - Ensure Railway service is running

2. **MT5 Connection Issues**
   - Verify MT5 credentials
   - Check if MT5 terminal is allowed for algo trading
   - Ensure correct server name

3. **WebSocket Disconnections**
   - Check CORS configuration
   - Verify WebSocket URL in frontend
   - Monitor Railway logs

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn main:app --reload
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/ -v
```

### Test WebSocket
```python
import websockets
import asyncio

async def test_ws():
    async with websockets.connect("ws://localhost:8080/ws") as ws:
        msg = await ws.recv()
        print(msg)

asyncio.run(test_ws())
```

## 📈 Performance Optimization

1. **Database Indexing**
   - Indexes on symbol, timestamp columns
   - Composite indexes for frequent queries

2. **Redis Caching**
   - Cache ML predictions (5 min TTL)
   - Cache market data (1 min TTL)
   - Session data for WebSocket clients

3. **Connection Pooling**
   - PostgreSQL: 5 connections
   - Redis: Persistent connections
   - MT5: Single persistent connection

## 🔐 Security

1. **API Security**
   - CORS restricted to frontend domain
   - Environment variables for secrets
   - SSL/TLS for all connections

2. **Trading Security**
   - Position size limits
   - Daily trade limits
   - Minimum confidence thresholds

3. **Data Security**
   - Encrypted database connections
   - No sensitive data in logs
   - Secure WebSocket connections

## 📚 API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- GitHub Issues: [Create Issue](https://github.com/SageRam/ML-Trading-System/issues)
- Documentation: Check `/docs` endpoint
- Logs: Monitor Railway logs for errors

---

**Note**: This is a financial trading system. Always test thoroughly in demo environments before using with real funds. Trading involves substantial risk of loss.