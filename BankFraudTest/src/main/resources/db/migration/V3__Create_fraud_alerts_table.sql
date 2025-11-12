-- V3: Create fraud_alerts table
-- This table stores fraud detection results and alerts

CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    alert_type VARCHAR(50) NOT NULL,
    fraud_score DECIMAL(5, 2) NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    rules_triggered TEXT[],
    description TEXT,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT NOW(),
    reviewed_at TIMESTAMP,
    reviewed_by VARCHAR(100)
);

-- Indexes
CREATE INDEX idx_fraud_alerts_transaction_id ON fraud_alerts(transaction_id);
CREATE INDEX idx_fraud_alerts_customer_id ON fraud_alerts(customer_id);
CREATE INDEX idx_fraud_alerts_status ON fraud_alerts(status);
CREATE INDEX idx_fraud_alerts_risk_level ON fraud_alerts(risk_level);
CREATE INDEX idx_fraud_alerts_created_at ON fraud_alerts(created_at DESC);
CREATE INDEX idx_fraud_alerts_fraud_score ON fraud_alerts(fraud_score DESC);

-- Check constraints
ALTER TABLE fraud_alerts ADD CONSTRAINT chk_fraud_score 
    CHECK (fraud_score >= 0 AND fraud_score <= 100);

ALTER TABLE fraud_alerts ADD CONSTRAINT chk_fraud_risk_level 
    CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'));

ALTER TABLE fraud_alerts ADD CONSTRAINT chk_fraud_status 
    CHECK (status IN ('PENDING', 'CONFIRMED', 'FALSE_POSITIVE', 'UNDER_REVIEW'));

-- Comments
COMMENT ON TABLE fraud_alerts IS 'Fraud detection alerts and investigation results';
COMMENT ON COLUMN fraud_alerts.fraud_score IS 'Fraud probability score (0-100)';
COMMENT ON COLUMN fraud_alerts.rules_triggered IS 'Array of rule IDs that triggered this alert';
COMMENT ON COLUMN fraud_alerts.status IS 'Alert status: PENDING, CONFIRMED, FALSE_POSITIVE, UNDER_REVIEW';
