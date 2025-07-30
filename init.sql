# init.sql - Database initialization
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Trades table
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(4) NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    entry_price DECIMAL(12, 6) NOT NULL,
    exit_price DECIMAL(12, 6),
    stop_loss DECIMAL(12, 6) NOT NULL,
    take_profit_1 DECIMAL(12, 6) NOT NULL,
    take_profit_2 DECIMAL(12, 6),
    take_profit_3 DECIMAL(12, 6),
    position_size DECIMAL(12, 6) NOT NULL,
    pnl DECIMAL(12, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    pattern VARCHAR(100),
    confidence DECIMAL(5, 4),
    open_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    close_time TIMESTAMP WITH TIME ZONE,
    mt5_ticket BIGINT,
    slippage DECIMAL(6, 2),
    commission DECIMAL(10, 2),
    swap DECIMAL(10, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Signals table
CREATE TABLE signals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    entry_price DECIMAL(12, 6) NOT NULL,
    stop_loss DECIMAL(12, 6) NOT NULL,
    tp1 DECIMAL(12, 6) NOT NULL,
    tp2 DECIMAL(12, 6),
    tp3 DECIMAL(12, 6),
    direction VARCHAR(4) NOT NULL CHECK (direction IN ('BUY', 'SELL')),
    confidence DECIMAL(5, 4) NOT NULL,
    pattern VARCHAR(100),
    risk_reward_ratio DECIMAL(8, 2),
    expected_value DECIMAL(12, 2),
    market_context JSONB,
    news_impact DECIMAL(5, 2),
    status VARCHAR(20) DEFAULT 'pending',
    executed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics table
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 4) DEFAULT 0,
    total_pnl DECIMAL(12, 2) DEFAULT 0,
    avg_pnl DECIMAL(12, 2) DEFAULT 0,
    max_drawdown DECIMAL(12, 2) DEFAULT 0,
    sharpe_ratio DECIMAL(8, 4) DEFAULT 0,
    sortino_ratio DECIMAL(8, 4) DEFAULT 0,
    profit_factor DECIMAL(8, 4) DEFAULT 0,
    current_balance DECIMAL(15, 2) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date)
);

-- Market data table
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    open_price DECIMAL(12, 6) NOT NULL,
    high_price DECIMAL(12, 6) NOT NULL,
    low_price DECIMAL(12, 6) NOT NULL,
    close_price DECIMAL(12, 6) NOT NULL,
    volume BIGINT DEFAULT 0,
    technical_indicators JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, timeframe, timestamp)
);

-- Reports table
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_type VARCHAR(50) NOT NULL,
    period VARCHAR(20) NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    data JSONB NOT NULL,
    file_path VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System configuration table
CREATE TABLE system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(100) NOT NULL UNIQUE,
    value JSONB NOT NULL,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default configuration
INSERT INTO system_config (key, value, description) VALUES
('max_concurrent_trades', '3', 'Maximum number of concurrent trades'),
('enable_auto_trading', 'true', 'Enable automatic trade execution'),
('risk_per_trade', '2.0', 'Risk percentage per trade'),
('storage_location', 'local', 'Data storage location: local or cloud'),
('mt5_enabled', 'true', 'Enable MT5 integration'),
('report_frequency', 'daily', 'Report generation frequency');

-- Create indexes for better performance
CREATE INDEX idx_trades_symbol_time ON trades(symbol, open_time);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_signals_created_at ON signals(created_at);
CREATE INDEX idx_market_data_symbol_time ON market_data(symbol, timestamp);
CREATE INDEX idx_performance_date ON performance_metrics(date);

-- Create trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at 
    BEFORE UPDATE ON trades 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();