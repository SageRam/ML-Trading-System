// App.tsx - Enhanced Production React Application
import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar,
  AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import './App.css';

// Production URLs from your services
const API_URL = process.env.REACT_APP_API_URL || 'https://ml-trading-system-production.up.railway.app';
const WS_URL = process.env.REACT_APP_WS_URL || 'wss://ml-trading-system-production.up.railway.app/ws';
const VERCEL_API_URL = 'https://nextjs-frontend-emkk2qczj-sagetronixs-projects.vercel.app';

// Types
interface MLAnalysis {
  symbol: string;
  timestamp: string;
  price: {
    current: number;
    open: number;
    high: number;
    low: number;
    change: number;
    change_pct: number;
    volume: number;
  };
  technical_indicators: {
    sma_20: number;
    sma_50: number;
    ema_12: number;
    ema_26: number;
    rsi_14: number;
    rsi_9: number;
    macd: number;
    macd_signal: number;
    macd_histogram: number;
    stoch_k: number;
    stoch_d: number;
    atr_14: number;
    bb_upper: number;
    bb_middle: number;
    bb_lower: number;
    bb_width: number;
    adx: number;
    plus_di: number;
    minus_di: number;
    obv: number;
    volume_ratio: number;
  };
  patterns: { [key: string]: number };
  ml_prediction: {
    status: string;
    prediction?: number;
    signal?: string;
    probability?: {
      buy: number;
      sell: number;
    };
    confidence?: number;
    strength?: string;
    top_features?: { [key: string]: number };
  };
  sentiment: {
    current: {
      overall: number;
      confidence: number;
      sources: any;
      themes: string[];
      sentiment_distribution: {
        positive: number;
        neutral: number;
        negative: number;
      };
    };
    trend: {
      trend: string;
      change: number;
      current: number;
      average: number;
      volatility: number;
    };
  };
  market_regime: {
    type: string;
    confidence: number;
    scores: { [key: string]: number };
    characteristics: {
      trend_strength: number;
      volatility_ratio: number;
      price_momentum: number;
      volume_profile: string;
    };
  };
  risk_metrics: {
    volatility: {
      daily: number;
      annualized: number;
      current: number;
    };
    value_at_risk: {
      var_95: number;
      cvar_95: number;
    };
    drawdown: {
      max: number;
      current: number;
    };
    ratios: {
      sharpe: number;
      sortino: number;
      calmar: number;
    };
    risk_score: number;
  };
  signal: any;
  market_conditions: {
    volatility: number;
    trend_strength: number;
    liquidity: string;
  };
}

interface Signal {
  id: string;
  symbol: string;
  entry_price: number;
  stop_loss: number;
  take_profit_levels: number[];
  direction: string;
  confidence: number;
  pattern: string;
  risk_reward_ratio: number;
  market_context: {
    regime: string;
    volatility: number;
    trend_strength: number;
    volume_profile: string;
  };
  timestamp: string;
  metadata?: {
    scores: { [key: string]: number };
    composite_score: number;
    technical_levels: {
      support: number;
      resistance: number;
      pivot: number;
    };
  };
}

interface Trade {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL';
  entry_price: number;
  exit_price?: number;
  stop_loss: number;
  take_profit_1: number;
  position_size: number;
  pnl?: number;
  status: 'OPEN' | 'CLOSED' | 'STOPPED';
  pattern?: string;
  confidence?: number;
  open_time: string;
  close_time?: string;
  mt5_ticket?: number;
}

interface Performance {
  total_trades: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  current_balance: number;
  max_drawdown: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  profit_factor: number;
}

interface SystemConfig {
  max_concurrent_trades: number;
  enable_auto_trading: boolean;
  risk_per_trade: number;
  storage_location: string;
  mt5_enabled: boolean;
}

// Custom Hook for WebSocket with reconnection
const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        setReconnectAttempt(0);
        console.log('WebSocket connected to Railway backend');
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('WebSocket disconnected, attempting reconnect...');
        
        // Exponential backoff for reconnection
        const timeout = Math.min(1000 * Math.pow(2, reconnectAttempt), 30000);
        setTimeout(() => {
          setReconnectAttempt(prev => prev + 1);
          connect();
        }, timeout);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      setSocket(ws);
    };

    connect();

    return () => {
      socket?.close();
    };
  }, [url, reconnectAttempt]);

  return { socket, isConnected, lastMessage };
};

// API Service for Railway backend
class TradingAPI {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_URL;
  }

  async get(endpoint: string) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
      }
    });
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    return response.json();
  }

  async post(endpoint: string, data: any) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }
    return response.json();
  }

  // ML Agent specific endpoints
  async getMLAnalysis(symbol: string = 'USDJPY'): Promise<MLAnalysis> {
    return this.get(`/api/ml/analyze/${symbol}`);
  }

  async getMLPerformance(): Promise<any> {
    return this.get('/api/ml/performance');
  }

  async retrainModel(): Promise<any> {
    return this.post('/api/ml/retrain', {});
  }

  // Vercel API integration
  async getMarketDataFromVercel(symbol: string = 'USDJPY'): Promise<any> {
    const response = await fetch(`${VERCEL_API_URL}/api/market-data/${symbol}`, {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    if (!response.ok) {
      throw new Error(`Vercel API Error: ${response.statusText}`);
    }
    return response.json();
  }

  async sendSignalToVercel(signal: Signal): Promise<any> {
    const response = await fetch(`${VERCEL_API_URL}/api/signals`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(signal),
    });
    if (!response.ok) {
      throw new Error(`Vercel API Error: ${response.statusText}`);
    }
    return response.json();
  }
}

// Main App Component
const App: React.FC = () => {
  // State
  const [mlAnalysis, setMlAnalysis] = useState<MLAnalysis | null>(null);
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [selectedSymbol, setSelectedSymbol] = useState('USDJPY');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mlPerformance, setMlPerformance] = useState<any>(null);

  // API instance
  const api = new TradingAPI();

  // WebSocket connection
  const { socket, isConnected, lastMessage } = useWebSocket(WS_URL);

  // Symbols to monitor
  const symbols = ['USDJPY', 'EURUSD', 'GBPUSD'];

  // Fetch ML Analysis
  const fetchMLAnalysis = useCallback(async (symbol: string) => {
    try {
      setLoading(true);
      const analysis = await api.getMLAnalysis(symbol);
      setMlAnalysis(analysis);
      
      // If there's a signal, add it to signals list
      if (analysis.signal) {
        const signal: Signal = {
          id: `${Date.now()}`,
          symbol: analysis.symbol,
          entry_price: analysis.signal.entry_price,
          stop_loss: analysis.signal.stop_loss,
          take_profit_levels: analysis.signal.take_profit_levels,
          direction: analysis.signal.direction,
          confidence: analysis.signal.confidence,
          pattern: analysis.signal.pattern,
          risk_reward_ratio: analysis.signal.risk_reward_ratio,
          market_context: analysis.signal.market_context,
          timestamp: analysis.signal.timestamp,
          metadata: analysis.signal.metadata
        };
        
        setSignals(prev => [signal, ...prev.slice(0, 49)]);
        
        // Send signal to Vercel frontend
        try {
          await api.sendSignalToVercel(signal);
        } catch (err) {
          console.error('Failed to send signal to Vercel:', err);
        }
      }
      
      setError(null);
    } catch (err: any) {
      setError(err.message);
      console.error('Error fetching ML analysis:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch ML Performance metrics
  const fetchMLPerformance = useCallback(async () => {
    try {
      const perfData = await api.getMLPerformance();
      setMlPerformance(perfData);
    } catch (err: any) {
      console.error('Error fetching ML performance:', err);
    }
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'ml_analysis_update':
          setMlAnalysis(lastMessage.data);
          break;
        case 'new_signal':
          setSignals(prev => [lastMessage.data, ...prev.slice(0, 49)]);
          break;
        case 'trade_executed':
          setTrades(prev => [lastMessage.data, ...prev.slice(0, 99)]);
          break;
        case 'ml_retrain_complete':
          fetchMLPerformance();
          break;
        default:
          break;
      }
    }
  }, [lastMessage, fetchMLPerformance]);

  // Initial data fetch
  useEffect(() => {
    fetchMLAnalysis(selectedSymbol);
    fetchMLPerformance();
    
    // Refresh ML analysis every 15 minutes (matching ML agent interval)
    const interval = setInterval(() => {
      fetchMLAnalysis(selectedSymbol);
    }, 900000); // 15 minutes
    
    return () => clearInterval(interval);
  }, [selectedSymbol, fetchMLAnalysis, fetchMLPerformance]);

  // Utility functions
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  const formatDateTime = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  // Retrain ML model
  const handleRetrainModel = async () => {
    try {
      setLoading(true);
      await api.retrainModel();
      setError('Model retraining initiated. This may take several minutes.');
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const prepareIndicatorData = () => {
    if (!mlAnalysis) return [];
    
    const indicators = mlAnalysis.technical_indicators;
    return [
      { name: 'RSI', value: indicators.rsi_14, max: 100 },
      { name: 'ADX', value: indicators.adx, max: 100 },
      { name: 'Stoch K', value: indicators.stoch_k, max: 100 },
      { name: 'Volume Ratio', value: indicators.volume_ratio * 100, max: 200 },
    ];
  };

  const prepareSentimentData = () => {
    if (!mlAnalysis) return [];
    
    const dist = mlAnalysis.sentiment.current.sentiment_distribution;
    return [
      { name: 'Positive', value: dist.positive, color: '#10b981' },
      { name: 'Neutral', value: dist.neutral, color: '#6b7280' },
      { name: 'Negative', value: dist.negative, color: '#ef4444' },
    ];
  };

  const prepareRiskMetricsData = () => {
    if (!mlAnalysis) return [];
    
    const risk = mlAnalysis.risk_metrics;
    return [
      { metric: 'Volatility', daily: risk.volatility.daily * 100, annualized: risk.volatility.annualized * 100 },
      { metric: 'VaR 95%', daily: Math.abs(risk.value_at_risk.var_95) * 100, annualized: Math.abs(risk.value_at_risk.cvar_95) * 100 },
      { metric: 'Drawdown', daily: Math.abs(risk.drawdown.current) * 100, annualized: Math.abs(risk.drawdown.max) * 100 },
    ];
  };

  if (loading && !mlAnalysis) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-gray-400">Connecting to ML Trading System...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 shadow-lg border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-white">ML Trading System</h1>
              <div className="ml-4 flex items-center space-x-2">
                <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm text-gray-400">
                  {isConnected ? 'Connected to Railway' : 'Disconnected'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-6">
              {/* Symbol Selector */}
              <select
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="bg-gray-700 text-white px-3 py-1 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
              >
                {symbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>

              {mlAnalysis && (
                <div className="text-sm">
                  <span className="text-gray-400">Current Price: </span>
                  <span className="font-mono text-white">{mlAnalysis.price.current.toFixed(5)}</span>
                  <span className={`ml-2 ${mlAnalysis.price.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {mlAnalysis.price.change >= 0 ? '▲' : '▼'} {formatPercentage(mlAnalysis.price.change_pct / 100)}
                  </span>
                </div>
              )}

              {/* Retrain Button */}
              <button
                onClick={handleRetrainModel}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors"
                disabled={loading}
              >
                Retrain Model
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-900 border border-red-700 text-red-100 px-4 py-3 mx-4 mt-4 rounded">
          <span className="block sm:inline">{error}</span>
          <button
            className="float-right font-bold text-red-100 hover:text-white"
            onClick={() => setError(null)}
          >
            ×
          </button>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="bg-gray-800 shadow-sm border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'ML Dashboard' },
              { id: 'analysis', label: 'Technical Analysis' },
              { id: 'signals', label: 'Trading Signals' },
              { id: 'performance', label: 'ML Performance' },
              { id: 'risk', label: 'Risk Analytics' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-600'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {/* ML Dashboard Tab */}
        {activeTab === 'dashboard' && mlAnalysis && (
          <div className="space-y-6">
            {/* ML Prediction Card */}
            {mlAnalysis.ml_prediction.status === 'success' && (
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-xl font-bold mb-4 text-white">ML Prediction</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="text-center">
                    <div className={`text-5xl font-bold mb-2 ${
                      mlAnalysis.ml_prediction.signal === 'BUY' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {mlAnalysis.ml_prediction.signal}
                    </div>
                    <div className="text-gray-400">Signal Direction</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-5xl font-bold mb-2 text-blue-400">
                      {formatPercentage(mlAnalysis.ml_prediction.confidence || 0)}
                    </div>
                    <div className="text-gray-400">Confidence</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="text-5xl font-bold mb-2 text-yellow-400">
                      {mlAnalysis.ml_prediction.strength}
                    </div>
                    <div className="text-gray-400">Signal Strength</div>
                  </div>
                </div>

                {/* Probability Bars */}
                {mlAnalysis.ml_prediction.probability && (
                  <div className="mt-6 space-y-3">
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-green-400">Buy Probability</span>
                        <span className="text-green-400">{formatPercentage(mlAnalysis.ml_prediction.probability.buy)}</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-4">
                        <div 
                          className="bg-green-500 h-4 rounded-full transition-all duration-500"
                          style={{ width: `${mlAnalysis.ml_prediction.probability.buy * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-red-400">Sell Probability</span>
                        <span className="text-red-400">{formatPercentage(mlAnalysis.ml_prediction.probability.sell)}</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-4">
                        <div 
                          className="bg-red-500 h-4 rounded-full transition-all duration-500"
                          style={{ width: `${mlAnalysis.ml_prediction.probability.sell * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Market Regime & Sentiment Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Market Regime */}
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-white">Market Regime</h3>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-blue-400 mb-2">
                      {mlAnalysis.market_regime.type.replace('_', ' ').toUpperCase()}
                    </div>
                    <div className="text-gray-400">
                      Confidence: {formatPercentage(mlAnalysis.market_regime.confidence)}
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    {Object.entries(mlAnalysis.market_regime.scores).map(([regime, score]) => (
                      <div key={regime} className="flex justify-between">
                        <span className="text-gray-400 capitalize">{regime.replace('_', ' ')}</span>
                        <div className="w-32 bg-gray-700 rounded-full h-2 ml-2">
                          <div 
                            className="bg-blue-500 h-2 rounded-full"
                            style={{ width: `${score * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Sentiment Analysis */}
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-white">Sentiment Analysis</h3>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className={`text-3xl font-bold mb-2 ${
                      mlAnalysis.sentiment.current.overall > 0 ? 'text-green-400' : 
                      mlAnalysis.sentiment.current.overall < 0 ? 'text-red-400' : 'text-gray-400'
                    }`}>
                      {mlAnalysis.sentiment.current.overall > 0 ? 'POSITIVE' : 
                       mlAnalysis.sentiment.current.overall < 0 ? 'NEGATIVE' : 'NEUTRAL'}
                    </div>
                    <div className="text-gray-400">
                      Score: {mlAnalysis.sentiment.current.overall.toFixed(3)}
                    </div>
                  </div>

                  {/* Sentiment Distribution Pie Chart */}
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={prepareSentimentData()}
                          cx="50%"
                          cy="50%"
                          innerRadius={40}
                          outerRadius={60}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {prepareSentimentData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value: number) => formatPercentage(value)} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="text-sm text-gray-400">
                    Trend: <span className={`font-semibold ${
                      mlAnalysis.sentiment.trend.trend === 'improving' ? 'text-green-400' :
                      mlAnalysis.sentiment.trend.trend === 'deteriorating' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>{mlAnalysis.sentiment.trend.trend}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Technical Indicators Overview */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-white">Key Technical Indicators</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={prepareIndicatorData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="name" tick={{ fill: '#9CA3AF' }} />
                    <YAxis tick={{ fill: '#9CA3AF' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Bar dataKey="value" fill="#3B82F6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* Technical Analysis Tab */}
        {activeTab === 'analysis' && mlAnalysis && (
          <div className="space-y-6">
            {/* Indicators Grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Object.entries(mlAnalysis.technical_indicators).map(([key, value]) => (
                <div key={key} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <div className="text-gray-400 text-sm uppercase">{key.replace(/_/g, ' ')}</div>
                  <div className="text-xl font-mono text-white mt-1">
                    {typeof value === 'number' ? value.toFixed(4) : value}
                  </div>
                </div>
              ))}
            </div>

            {/* Patterns Detected */}
            {Object.keys(mlAnalysis.patterns).length > 0 && (
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-white">Detected Patterns</h3>
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                  {Object.entries(mlAnalysis.patterns).map(([pattern, confidence]) => (
                    <div key={pattern} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                      <span className="text-gray-300 text-sm">{pattern.replace(/_/g, ' ')}</span>
                      <span className={`font-semibold ${
                        confidence > 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {(confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Top Features from ML Model */}
            {mlAnalysis.ml_prediction.top_features && (
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-white">Top ML Features</h3>
                <div className="space-y-3">
                  {Object.entries(mlAnalysis.ml_prediction.top_features).map(([feature, importance]) => (
                    <div key={feature}>
                      <div className="flex justify-between mb-1">
                        <span className="text-gray-400">{feature.replace(/_/g, ' ')}</span>
                        <span className="text-gray-300">{(importance * 100).toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${importance * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Trading Signals Tab */}
        {activeTab === 'signals' && (
          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
              <h2 className="text-xl font-bold mb-4 text-white">Recent Trading Signals</h2>
              {signals.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-700">
                    <thead className="bg-gray-900">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Time</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Direction</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Entry</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Stop Loss</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Take Profit</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Confidence</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Pattern</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">R:R</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-700">
                      {signals.map((signal) => (
                        <tr key={signal.id} className="hover:bg-gray-700 transition-colors">
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            {new Date(signal.timestamp).toLocaleTimeString()}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                            {signal.symbol}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm">
                            <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                              signal.direction.includes('BUY') 
                                ? 'bg-green-900 text-green-300' 
                                : 'bg-red-900 text-red-300'
                            }`}>
                              {signal.direction}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300 font-mono">
                            {signal.entry_price.toFixed(5)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-red-400 font-mono">
                            {signal.stop_loss.toFixed(5)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-green-400 font-mono">
                            {signal.take_profit_levels[0].toFixed(5)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            {formatPercentage(signal.confidence)}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            {signal.pattern.replace(/_/g, ' ')}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                            1:{signal.risk_reward_ratio.toFixed(2)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p className="text-gray-400 text-center py-8">No signals generated yet. The ML model analyzes markets every 15 minutes.</p>
              )}
            </div>
          </div>
        )}

        {/* ML Performance Tab */}
        {activeTab === 'performance' && mlPerformance && (
          <div className="space-y-6">
            {/* Model Status */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
              <h2 className="text-xl font-bold mb-4 text-white">ML Model Status</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className={`text-3xl font-bold mb-2 ${
                    mlPerformance.ml_model?.is_trained ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {mlPerformance.ml_model?.is_trained ? 'TRAINED' : 'NOT TRAINED'}
                  </div>
                  <div className="text-gray-400">Model Status</div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold mb-2 text-blue-400">
                    {mlPerformance.ml_model?.features_count || 0}
                  </div>
                  <div className="text-gray-400">Features</div>
                </div>
                
                <div className="text-center">
                  <div className="text-3xl font-bold mb-2 text-purple-400">
                    {mlPerformance.ml_model?.models_count || 0}
                  </div>
                  <div className="text-gray-400">Models in Ensemble</div>
                </div>
              </div>

              {mlPerformance.ml_model?.last_performance && (
                <div className="mt-6 space-y-3">
                  <h3 className="text-lg font-semibold text-white">Last Training Performance</h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {Object.entries(mlPerformance.ml_model.last_performance.scores).map(([model, scores]: [string, any]) => (
                      <div key={model} className="bg-gray-700 rounded p-3">
                        <div className="text-sm text-gray-400 uppercase">{model}</div>
                        <div className="text-xl font-mono text-white mt-1">
                          {(scores.accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-500">Accuracy</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* System Performance */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-white">System Performance</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Signals Generated</span>
                  <span className="text-white font-semibold">{mlPerformance.signals?.total_generated || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Signals (Last 24h)</span>
                  <span className="text-white font-semibold">{mlPerformance.signals?.last_24h || 0}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Redis Connection</span>
                  <span className={`font-semibold ${
                    mlPerformance.system?.redis_connected ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {mlPerformance.system?.redis_connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">PostgreSQL Connection</span>
                  <span className={`font-semibold ${
                    mlPerformance.system?.db_connected ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {mlPerformance.system?.db_connected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
              </div>

              {mlPerformance.signals?.distribution && (
                <div className="mt-6">
                  <h4 className="text-sm text-gray-400 mb-3">Signal Distribution</h4>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(mlPerformance.signals.distribution).map(([type, count]) => (
                      <div key={type} className="bg-gray-700 rounded p-3 text-center">
                        <div className="text-xs text-gray-400">{type}</div>
                        <div className="text-lg font-semibold text-white">{count}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Risk Analytics Tab */}
        {activeTab === 'risk' && mlAnalysis && (
          <div className="space-y-6">
            {/* Risk Metrics Overview */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
              <h2 className="text-xl font-bold mb-4 text-white">Risk Metrics</h2>
              
              {/* Risk Score Gauge */}
              <div className="text-center mb-6">
                <div className="text-5xl font-bold text-white mb-2">
                  {(mlAnalysis.risk_metrics.risk_score * 100).toFixed(0)}%
                </div>
                <div className="text-gray-400">Overall Risk Score</div>
                <div className="w-full bg-gray-700 rounded-full h-4 mt-3 max-w-md mx-auto">
                  <div 
                    className={`h-4 rounded-full transition-all duration-500 ${
                      mlAnalysis.risk_metrics.risk_score < 0.3 ? 'bg-green-500' :
                      mlAnalysis.risk_metrics.risk_score < 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${mlAnalysis.risk_metrics.risk_score * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Risk Metrics Chart */}
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={prepareRiskMetricsData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="metric" tick={{ fill: '#9CA3AF' }} />
                    <YAxis tick={{ fill: '#9CA3AF' }} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                      formatter={(value: number) => `${value.toFixed(2)}%`}
                    />
                    <Bar dataKey="daily" fill="#3B82F6" name="Daily" />
                    <Bar dataKey="annualized" fill="#8B5CF6" name="Annualized" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Detailed Risk Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Volatility Metrics */}
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-white">Volatility Analysis</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Daily Volatility</span>
                    <span className="text-white font-mono">{formatPercentage(mlAnalysis.risk_metrics.volatility.daily)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Annualized Volatility</span>
                    <span className="text-white font-mono">{formatPercentage(mlAnalysis.risk_metrics.volatility.annualized)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Current Volatility</span>
                    <span className="text-white font-mono">{formatPercentage(mlAnalysis.risk_metrics.volatility.current)}</span>
                  </div>
                </div>
              </div>

              {/* Risk Ratios */}
              <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
                <h3 className="text-lg font-semibold mb-4 text-white">Risk-Adjusted Returns</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sharpe Ratio</span>
                    <span className="text-white font-mono">{mlAnalysis.risk_metrics.ratios.sharpe.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Sortino Ratio</span>
                    <span className="text-white font-mono">{mlAnalysis.risk_metrics.ratios.sortino.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Calmar Ratio</span>
                    <span className="text-white font-mono">{mlAnalysis.risk_metrics.ratios.calmar.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Value at Risk */}
            <div className="bg-gray-800 rounded-lg shadow-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4 text-white">Value at Risk (95% Confidence)</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-400">
                    {formatPercentage(Math.abs(mlAnalysis.risk_metrics.value_at_risk.var_95))}
                  </div>
                  <div className="text-gray-400 text-sm">VaR (95%)</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-500">
                    {formatPercentage(Math.abs(mlAnalysis.risk_metrics.value_at_risk.cvar_95))}
                  </div>
                  <div className="text-gray-400 text-sm">CVaR (95%)</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-400">
                    {formatPercentage(Math.abs(mlAnalysis.risk_metrics.drawdown.current))}
                  </div>
                  <div className="text-gray-400 text-sm">Current Drawdown</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-500">
                    {formatPercentage(Math.abs(mlAnalysis.risk_metrics.drawdown.max))}
                  </div>
                  <div className="text-gray-400 text-sm">Max Drawdown</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;