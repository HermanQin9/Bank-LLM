-- V1: Create transactions table
-- This table stores normalized transaction data from multiple banking sources

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    transaction_date TIMESTAMP NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    merchant_name VARCHAR(255),
    merchant_category VARCHAR(50),
    transaction_type VARCHAR(20),
    card_last_four VARCHAR(4),
    location_country VARCHAR(2),
    location_city VARCHAR(100),
    ip_address INET,
    device_fingerprint VARCHAR(255),
    is_online BOOLEAN DEFAULT TRUE,
    source_system VARCHAR(50) NOT NULL,
    raw_data JSONB,
    normalized_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date DESC);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE INDEX idx_transactions_source_system ON transactions(source_system);
CREATE INDEX idx_transactions_merchant_category ON transactions(merchant_category);
CREATE INDEX idx_transactions_created_at ON transactions(created_at DESC);

-- Index for JSON queries
CREATE INDEX idx_transactions_raw_data_gin ON transactions USING GIN(raw_data);

-- Comments
COMMENT ON TABLE transactions IS 'Normalized transaction data from multiple banking sources';
COMMENT ON COLUMN transactions.source_system IS 'Source banking system identifier (e.g., BANK_A, BANK_B)';
COMMENT ON COLUMN transactions.raw_data IS 'Original raw data in JSON format for audit trail';
