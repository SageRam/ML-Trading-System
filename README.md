# ML Trading System

A production-ready ML-powered trading system with real-time market analysis, automated signal generation, and comprehensive risk management.

## 🚀 Architecture Overview

The system consists of three main components:

1. **ML Agent (Railway)** - Python-based ML analytics engine
2. **Frontend (Vercel)** - React-based trading dashboard
3. **Data Storage** - PostgreSQL and Redis on Railway

### System Architecture
```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Vercel Frontend   │────▶│    Railway ML       │────▶│   Railway DBs       │
│   (React + TS)      │◀────│    Agent (Python)   │◀────│ (PostgreSQL+Redis)  │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                            │                            │
         │                            │                            │
         ▼                            ▼                            ▼
   Market Display              ML Predictions              Data Persistence
   Signal Alerts              Pattern Recognition          Model Storage
   Risk Analytics            Sentiment Analysis           Cache Layer
```

## 🛠️ Tech Stack

### ML Agent (Backend)
- **Language**: Python 3.9+
- **ML Framework**: scikit-learn with ensemble methods
- **Technical Analysis**: TA-Lib, pandas
- **Database**: PostgreSQL (Railway)
- **Cache**: Redis (Railway)
- **API**: FastAPI/ASGI
- **Deployment**: Railway

### Frontend
- **Framework**: React 18 with TypeScript
- **Charts**: Recharts
- **Styling**: Custom CSS with dark theme
- **WebSocket**: Real-time updates
- **Deployment**: Vercel

## 📊 Features

### ML Capabilities
- **Ensemble Learning**: Random Forest, Gradient Boosting, Voting Classifier
- **Technical Indicators**: 50+ indicators including RSI, MACD, Bollinger Bands
- **Pattern Recognition**: Candlestick patterns, chart patterns, harmonic patterns
- **Sentiment Analysis**: Multi-source news sentiment aggregation
- **Risk Management**: VaR, CVaR, Sharpe ratio, drawdown analysis

### Trading Features
- Real-time signal generation for USDJPY, EURUSD, GBPUSD
- Automated pattern detection
- Market regime classification
- Multi-timeframe analysis
- Risk-adjusted position sizing

### Dashboard Features
- Live ML predictions with confidence scores
- Technical indicator visualization
- Risk metrics monitoring
- Signal history tracking
- Performance analytics

## 🚀 Deployment Guide

### Prerequisites
- GitHub account
- Railway account
- Vercel account
- Node.js 16+ (for local development)
- Python 3.9+ (for local ML development)

### Step 1: Clone Repository
```bash
git clone https://github.com/SageRam/ML-Trading-System.git
cd ML-Trading-System
```

### Step 2: Deploy ML Agent to Railway

1. Create a new project on Railway
2. Connect your GitHub repository
3. Add PostgreSQL and Redis services
4. Set environment variables:
   ```
   DATABASE_URL=postgresql://postgres:xPgernhZHDqzwMZZSusJrpPuUxZyUYoR@ballast.proxy.rlwy.net:56300/railway
   REDIS_URL=redis://default:vanoEzlkdWoeXeSbOYIrtagILrJaugmb@yamabiko.proxy.rlwy.net:22322
   API_URL=https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app
   NEWS_API_KEY=your_key_here
   ALPHA_VANTAGE_KEY=your_key_here
   FINNHUB_KEY=your_key_here
   ```
5. Deploy using:
   ```bash
   railway up
   ```

### Step 3: Deploy Frontend to Vercel

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Deploy to Vercel:
   ```bash
   cd frontend
   npm install
   vercel
   ```

3. Set environment variables in Vercel dashboard:
   ```
   REACT_APP_API_URL=https://your-railway-app.up.railway.app
   REACT_APP_WS_URL=wss://your-railway-app.up.railway.app/ws
   REACT_APP_VERCEL_API_URL=https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app
   ```

### Step 4: Configure Production URLs

Update the following files with your production URLs:

1. **frontend/.env**
   ```
   REACT_APP_API_URL=https://your-railway-app.up.railway.app
   REACT_APP_WS_URL=wss://your-railway-app.up.railway.app/ws
   ```

2. **config.py**
   ```python
   DATABASE_URL = "your-railway-postgres-url"
   REDIS_URL = "your-railway-redis-url"
   API_URL = "your-vercel-frontend-url"
   ```

## 📝 Configuration

### ML Agent Configuration
Edit `config.py` to adjust:
- Model update intervals
- Trading symbols
- Risk parameters
- Signal thresholds

### Frontend Configuration
Edit `frontend/.env` to configure:
- API endpoints
- WebSocket URLs
- Feature flags

## 🔧 Local Development

### ML Agent
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python ml_agent_enhanced.py
```

### Frontend
```bash
# Install dependencies
cd frontend
npm install

# Run development server
npm start
```

## 📊 API Endpoints

### ML Agent Endpoints
- `GET /api/ml/analyze/{symbol}` - Get ML analysis for symbol
- `GET /api/ml/performance` - Get ML model performance metrics
- `POST /api/ml/retrain` - Trigger model retraining
- `GET /api/signals` - Get recent trading signals
- `GET /api/market-data/{symbol}` - Get market data

### WebSocket Events
- `ml_analysis_update` - Real-time ML analysis updates
- `new_signal` - New trading signal generated
- `trade_executed` - Trade execution notification
- `ml_retrain_complete` - Model retraining completed

## 🔐 Security

- All production credentials should be stored as environment variables
- Use Railway's built-in SSL for database connections
- Enable CORS only for your Vercel domain
- Implement rate limiting for API endpoints

## 📈 Performance Optimization

- Redis caching for frequently accessed data
- Efficient numpy operations for technical indicators
- WebSocket for real-time updates
- Lazy loading for frontend components
- CDN for static assets

## 🐛 Troubleshooting

### Common Issues

1. **ML Agent not connecting to databases**
   - Verify DATABASE_URL and REDIS_URL are correct
   - Check Railway service status

2. **Frontend WebSocket disconnection**
   - Ensure WS_URL uses wss:// for production
   - Check CORS settings on Railway

3. **Model training failures**
   - Verify sufficient memory allocation on Railway
   - Check data quality and availability

## 📚 Documentation

- [ML Agent Documentation](./docs/ml-agent.md)
- [Frontend Documentation](./docs/frontend.md)
- [API Reference](./docs/api.md)
- [Trading Strategy Guide](./docs/strategy.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [TA-Lib](https://github.com/mrjbq7/ta-lib) for technical analysis
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [Railway](https://railway.app/) for hosting
- [Vercel](https://vercel.com/) for frontend deployment

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review closed issues for solutions

---

**Disclaimer**: This software is for educational purposes only. Trading involves substantial risk of loss. Always do your own research and consider your financial situation carefully before trading.
