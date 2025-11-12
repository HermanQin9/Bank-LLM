-- V2: Create customers table
-- This table stores customer master data and risk profiles

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(20),
    account_created_date DATE,
    risk_level VARCHAR(20) DEFAULT 'LOW',
    total_transactions INTEGER DEFAULT 0,
    lifetime_value DECIMAL(15, 2) DEFAULT 0.00,
    last_transaction_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_risk_level ON customers(risk_level);
CREATE INDEX idx_customers_last_transaction_date ON customers(last_transaction_date DESC);

-- Check constraint for risk level
ALTER TABLE customers ADD CONSTRAINT chk_risk_level 
    CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'));

-- Comments
COMMENT ON TABLE customers IS 'Customer master data with risk profiling';
COMMENT ON COLUMN customers.risk_level IS 'Risk classification: LOW, MEDIUM, HIGH, CRITICAL';
COMMENT ON COLUMN customers.lifetime_value IS 'Total transaction value across customer lifetime';
