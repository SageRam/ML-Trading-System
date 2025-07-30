// App.tsx - Main React Application
import React, { useState, useEffect, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar,
  AreaChart, Area
} from 'recharts';
import './App.css';

// Types
interface Signal {
  id: string;
  symbol: string;
  entry_price: number;
  stop_loss: number;
  tp1: number;
  tp2?: number;
  tp3?: number;
  direction: 'BUY' | 'SELL';
  confidence: number;
  pattern: string;
  status: 'pending' | 'executed' | 'rejected';
  created_at: string;
  executed_at?: string;
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

interface Position {
  ticket: number;
  symbol: string;
  type: 'BUY' | 'SELL';
  volume: number;
  price_open: number;
  price_current: number;
  sl: number;
  tp: number;
  profit: number;
  time: string;
}

// Custom Hook for WebSocket
const useWebSocket = (url: string) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<any>(null);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
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
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    setSocket(ws);

    return () => {
      ws.close();
    };
  }, [url]);

  return { socket, isConnected, lastMessage };
};

// API Service
class TradingAPI {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:8080') {
    this.baseUrl = baseUrl;
  }

  async get(endpoint: string) {
    const response = await fetch(`${this.baseUrl}${endpoint}`);
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

  async put(endpoint: string, data: any) {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      method: 'PUT',
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

  // Specific API methods
  async getPerformance(): Promise<Performance> {
    return this.get('/api/performance');
  }

  async getSignals(): Promise<{ signals: Signal[]; count: number }> {
    return this.get('/api/signals');
  }

  async getTrades(): Promise<{ trades: Trade[]; count: number }> {
    return this.get('/api/trades');
  }

  async getPositions(): Promise<{ positions: Position[]; count: number }> {
    return this.get('/api/positions');
  }

  async getConfig(): Promise<SystemConfig> {
    return this.get('/api/config');
  }

  async updateConfig(config: SystemConfig): Promise<any> {
    return this.put('/api/config', config);
  }

  async generateReport(reportType: string, period: string): Promise<any> {
    return this.get(`/api/reports/${reportType}?period=${period}`);
  }

  async getReports(): Promise<any> {
    return this.get('/api/reports');
  }

  async getMarketData(symbol: string = 'USDJPY'): Promise<any> {
    return this.get(`/api/market-data/${symbol}`);
  }
}

// Main App Component
const App: React.FC = () => {
  // State
  const [performance, setPerformance] = useState<Performance | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [marketData, setMarketData] = useState<any>(null);
  const [reports, setReports] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // API instance
  const api = new TradingAPI(process.env.REACT_APP_API_URL);

  // WebSocket connection
  const { socket, isConnected, lastMessage } = useWebSocket(
    process.env.REACT_APP_WS_URL || 'ws://localhost:8080/ws'
  );

  // Fetch data functions
  const fetchAllData = useCallback(async () => {
    try {
      setLoading(true);
      const [perfData, signalsData, tradesData, positionsData, configData] = await Promise.all([
        api.getPerformance(),
        api.getSignals(),
        api.getTrades(),
        api.getPositions(),
        api.getConfig(),
      ]);

      setPerformance(perfData);
      setSignals(signalsData.signals);
      setTrades(tradesData.trades);
      setPositions(positionsData.positions);
      setConfig(configData);
      setError(null);
    } catch (err: any) {
      setError(err.message);
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'new_signal':
          setSignals(prev => [lastMessage.data, ...prev.slice(0, 49)]);
          break;
        case 'trade_executed':
          fetchAllData(); // Refresh all data when trade is executed
          break;
        case 'market_update':
          setMarketData(lastMessage.data);
          break;
        default:
          break;
      }
    }
  }, [lastMessage, fetchAllData]);

  // Initial data fetch
  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [fetchAllData]);

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

  // Update configuration
  const updateConfiguration = async (newConfig: Partial<SystemConfig>) => {
    if (!config) return;
    
    try {
      const updatedConfig = { ...config, ...newConfig };
      await api.updateConfig(updatedConfig);
      setConfig(updatedConfig);
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Generate report
  const generateReport = async (reportType: string, period: string) => {
    try {
      await api.generateReport(reportType, period);
      // Refresh reports list
      const reportsData = await api.getReports();
      setReports(reportsData.reports);
    } catch (err: any) {
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-gray-900">Advanced Trading System</h1>
              <div className="ml-4 flex items-center space-x-2">
                <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></div>
                <span className="text-sm text-gray-600">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {marketData && (
                <div className="text-sm">
                  <span className="text-gray-600">USDJPY: </span>
                  <span className="font-mono">{marketData.price?.toFixed(5)}</span>
                </div>
              )}
              
              {performance && (
                <div className="text-sm">
                  <span className="text-gray-600">Balance: </span>
                  <span className="font-semibold">{formatCurrency(performance.current_balance)}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 mx-4 mt-4 rounded">
          <span className="block sm:inline">{error}</span>
          <button
            className="float-right font-bold text-red-700 hover:text-red-900"
            onClick={() => setError(null)}
          >
            ×
          </button>
        </div>
      )}

      {/* Navigation Tabs */}
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'dashboard', label: 'Dashboard' },
              { id: 'signals', label: 'Signals' },
              { id: 'trades', label: 'Trades' },
              { id: 'positions', label: 'Positions' },
              { id: 'reports', label: 'Reports' },
              { id: 'settings', label: 'Settings' },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-4 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
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
        {/* Dashboard Tab */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6">
            {performance && (
              <>
                {/* Performance Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <div className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="p-5">
                      <div className="flex items-center">
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 bg-green-100 rounded-md flex items-center justify-center">
                            <span className="text-green-600 font-bold">$</span>
                          </div>
                        </div>
                        <div className="ml-5 w-0 flex-1">
                          <dl>
                            <dt className="text-sm font-medium text-gray-500 truncate">Total P&L</dt>
                            <dd className={`text-lg font-medium ${performance.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {formatCurrency(performance.total_pnl)}
                            </dd>
                          </dl>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="p-5">
                      <div className="flex items-center">