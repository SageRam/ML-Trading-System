# config.py - Configuration for ML Agent deployment

import os
from typing import Dict, Any

class Config:
    """Configuration for ML Trading Agent"""
    
    # Production Services
    DATABASE_URL = "postgresql://postgres:xPgernhZHDqzwMZZSusJrpPuUxZyUYoR@ballast.proxy.rlwy.net:56300/railway"
    REDIS_URL = "redis://default:vanoEzlkdWoeXeSbOYIrtagILrJaugmb@yamabiko.proxy.rlwy.net:22322"
    API_URL = "https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app"
    
    # Model Configuration
    MODEL_UPDATE_INTERVAL = 3600  # 1 hour
    MODEL_DIR = "models"
    
    # Trading Configuration
    SYMBOLS = ["USDJPY", "EURUSD", "GBPUSD"]
    DEFAULT_TIMEFRAME = "M15"
    SIGNAL_INTERVAL = 900  # 15 minutes
    
    # Risk Management
    MAX_RISK_PER_TRADE = 0.02  # 2%
    DEFAULT_POSITION_SIZE = 0.01  # 0.01 lots
    
    # API Keys (set these as environment variables)
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
    FINNHUB_KEY = os.getenv('FINNHUB_KEY', '')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = 'ml_agent.log'
    
    # Performance
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    REQUEST_TIMEOUT = 30
    
    # Redis Cache Configuration
    CACHE_TTL = {
        'market_data': 300,      # 5 minutes
        'analysis': 300,         # 5 minutes
        'sentiment': 3600,       # 1 hour
        'model_metadata': 86400  # 24 hours
    }
    
    @classmethod
    def get_redis_config(cls) -> Dict[str, Any]:
        """Get Redis configuration for production"""
        return {
            'decode_responses': True,
            'socket_connect_timeout': 30,
            'socket_timeout': 30,
            'retry_on_timeout': True,
            'retry_on_error': [ConnectionError, TimeoutError]
        }
    
    @classmethod
    def get_db_config(cls) -> Dict[str, Any]:
        """Get PostgreSQL configuration for Railway"""
        return {
            'pool_size': 5,
            'max_overflow': 10,
            'pool_timeout': 30,
            'pool_recycle': 1800,
            'connect_args': {
                'sslmode': 'require',
                'connect_timeout': 30
            }
        }
    
    @classmethod
    def get_api_headers(cls) -> Dict[str, str]:
        """Get headers for API requests to Vercel"""
        return {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ML-Trading-Agent/1.0'
        }

# Create requirements.txt content
REQUIREMENTS = """
# Core Dependencies
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
joblib==1.3.1
scipy==1.11.1

# Technical Analysis
TA-Lib==0.4.28
yfinance==0.2.28

# Database and Caching
redis==5.0.0
sqlalchemy==2.0.19
psycopg2-binary==2.9.6
asyncpg==0.28.0

# API and Web
requests==2.31.0
aiohttp==3.8.5
fastapi==0.103.0
uvicorn==0.23.2

# Sentiment Analysis
textblob==0.17.1
nltk==3.8.1

# MetaTrader (optional)
# MetaTrader5==5.0.45

# Utilities
python-dotenv==1.0.0
pydantic==2.3.0
typing-extensions==4.7.1

# Logging and Monitoring
loguru==0.7.0

# Development
pytest==7.4.0
pytest-asyncio==0.21.1
black==23.7.0
"""

# Create railway.json for deployment
RAILWAY_CONFIG = """
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python ml_agent_enhanced.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  },
  "variables": {
    "DATABASE_URL": "${{Postgres.DATABASE_URL}}",
    "REDIS_URL": "${{Redis.REDIS_URL}}",
    "API_URL": "https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app",
    "PYTHONUNBUFFERED": "1"
  }
}
"""

# Create Procfile for Railway
PROCFILE = """
worker: python ml_agent_enhanced.py
"""

# Create .env.example
ENV_EXAMPLE = """
# Production Services (already configured in code)
DATABASE_URL=postgresql://postgres:xPgernhZHDqzwMZZSusJrpPuUxZyUYoR@ballast.proxy.rlwy.net:56300/railway
REDIS_URL=redis://default:vanoEzlkdWoeXeSbOYIrtagILrJaugmb@yamabiko.proxy.rlwy.net:22322
API_URL=https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app

# API Keys (add your own)
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
FINNHUB_KEY=your_finnhub_key_here

# Optional Configuration
LOG_LEVEL=INFO
MODEL_UPDATE_INTERVAL=3600
"""

if __name__ == "__main__":
    # Create deployment files
    with open('requirements.txt', 'w') as f:
        f.write(REQUIREMENTS.strip())
    
    with open('railway.json', 'w') as f:
        f.write(RAILWAY_CONFIG.strip())
    
    with open('Procfile', 'w') as f:
        f.write(PROCFILE.strip())
    
    with open('.env.example', 'w') as f:
        f.write(ENV_EXAMPLE.strip())
    
    print("Deployment configuration files created successfully!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env and add your API keys")
    print("2. Commit all files to your GitHub repository")
    print("3. Deploy to Railway using the GitHub integration")
    print("4. Set environment variables in Railway dashboard")