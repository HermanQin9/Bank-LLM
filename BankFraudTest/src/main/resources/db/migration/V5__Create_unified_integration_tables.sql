-- Customer Profiles Table (stores LLM-extracted business information)
-- This bridges document intelligence with transaction monitoring

CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id VARCHAR(50) PRIMARY KEY,
    business_type VARCHAR(100) NOT NULL,
    expected_monthly_volume DECIMAL(15, 2),
    expected_min_amount DECIMAL(15, 2),
    expected_max_amount DECIMAL(15, 2),
    geographic_scope JSONB,  -- ["USA", "Canada", "Europe"]
    risk_indicators JSONB,   -- ["PEP", "High-risk jurisdiction"]
    kyc_document_source TEXT,
    confidence_score DECIMAL(3, 2),  -- LLM extraction confidence (0-1)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Link to existing customers table
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Create indexes for performance
CREATE INDEX idx_customer_profiles_business_type ON customer_profiles(business_type);
CREATE INDEX idx_customer_profiles_risk ON customer_profiles USING GIN(risk_indicators);

-- Transaction Alerts Table (stores anomalies detected by unified system)
CREATE TABLE IF NOT EXISTS transaction_alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) NOT NULL,
    customer_id VARCHAR(50) NOT NULL,
    alert_type VARCHAR(50) NOT NULL,  -- 'AMOUNT_ANOMALY', 'GEO_MISMATCH', etc.
    severity VARCHAR(20) NOT NULL,    -- 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    deviation_details JSONB,
    supporting_evidence JSONB,  -- Array of document excerpts
    recommended_action VARCHAR(100),
    status VARCHAR(20) DEFAULT 'PENDING',  -- 'PENDING', 'REVIEWED', 'CLOSED'
    reviewed_by VARCHAR(50),
    review_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes
CREATE INDEX idx_alerts_customer ON transaction_alerts(customer_id);
CREATE INDEX idx_alerts_status ON transaction_alerts(status);
CREATE INDEX idx_alerts_severity ON transaction_alerts(severity);
CREATE INDEX idx_alerts_type ON transaction_alerts(alert_type);
CREATE INDEX idx_alerts_created ON transaction_alerts(created_at DESC);

-- Document Evidence Table (links documents to transactions/alerts)
CREATE TABLE IF NOT EXISTS document_evidence (
    evidence_id SERIAL PRIMARY KEY,
    alert_id VARCHAR(50),
    transaction_id VARCHAR(50),
    customer_id VARCHAR(50) NOT NULL,
    document_type VARCHAR(50),  -- 'KYC', 'CORRESPONDENCE', 'CONTRACT', 'SAR'
    document_path TEXT,
    excerpt TEXT,  -- Relevant excerpt extracted by LLM
    relevance_score DECIMAL(3, 2),  -- RAG similarity score
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (alert_id) REFERENCES transaction_alerts(alert_id) ON DELETE CASCADE,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes
CREATE INDEX idx_evidence_alert ON document_evidence(alert_id);
CREATE INDEX idx_evidence_customer ON document_evidence(customer_id);
CREATE INDEX idx_evidence_type ON document_evidence(document_type);

-- Compliance Reports Table (stores generated SAR/CTR reports)
CREATE TABLE IF NOT EXISTS compliance_reports (
    report_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    report_type VARCHAR(20) NOT NULL,  -- 'SAR', 'CTR', 'CDD'
    report_period_start DATE,
    report_period_end DATE,
    suspicious_transaction_count INTEGER,
    total_suspicious_amount DECIMAL(15, 2),
    report_content TEXT,  -- LLM-generated report
    filed_with_regulator BOOLEAN DEFAULT FALSE,
    filing_date DATE,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generated_by VARCHAR(50) DEFAULT 'AUTOMATED_SYSTEM',
    
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create indexes
CREATE INDEX idx_reports_customer ON compliance_reports(customer_id);
CREATE INDEX idx_reports_type ON compliance_reports(report_type);
CREATE INDEX idx_reports_filed ON compliance_reports(filed_with_regulator);
CREATE INDEX idx_reports_period ON compliance_reports(report_period_start, report_period_end);

-- View: Customer Risk Dashboard (combines profile + transactions + alerts)
CREATE OR REPLACE VIEW customer_risk_dashboard AS
SELECT 
    cp.customer_id,
    cp.business_type,
    cp.expected_monthly_volume,
    cp.risk_indicators,
    
    -- Transaction statistics
    COUNT(t.transaction_id) as total_transactions,
    COALESCE(SUM(t.amount), 0) as total_volume,
    COALESCE(AVG(t.amount), 0) as avg_transaction,
    
    -- Alert statistics
    COUNT(DISTINCT ta.alert_id) as total_alerts,
    COUNT(DISTINCT ta.alert_id) FILTER (WHERE ta.severity = 'HIGH' OR ta.severity = 'CRITICAL') as critical_alerts,
    COUNT(DISTINCT ta.alert_id) FILTER (WHERE ta.status = 'PENDING') as pending_alerts,
    
    -- Risk score (simple calculation)
    CASE 
        WHEN COUNT(DISTINCT ta.alert_id) FILTER (WHERE ta.severity = 'CRITICAL') > 0 THEN 'CRITICAL'
        WHEN COUNT(DISTINCT ta.alert_id) FILTER (WHERE ta.severity = 'HIGH') > 2 THEN 'HIGH'
        WHEN COUNT(DISTINCT ta.alert_id) > 5 THEN 'MEDIUM'
        ELSE 'LOW'
    END as overall_risk_level,
    
    -- Compliance status
    EXISTS(SELECT 1 FROM compliance_reports cr WHERE cr.customer_id = cp.customer_id AND cr.filed_with_regulator = TRUE) as sar_filed,
    
    cp.updated_at as profile_last_updated
FROM customer_profiles cp
LEFT JOIN transactions t ON t.customer_id = cp.customer_id 
    AND t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
LEFT JOIN transaction_alerts ta ON ta.customer_id = cp.customer_id
    AND ta.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY cp.customer_id, cp.business_type, cp.expected_monthly_volume, 
         cp.risk_indicators, cp.updated_at;

-- Function: Calculate transaction deviation from profile
CREATE OR REPLACE FUNCTION check_transaction_deviation(
    p_customer_id VARCHAR(50),
    p_amount DECIMAL(15, 2)
) RETURNS TABLE (
    is_anomaly BOOLEAN,
    deviation_type VARCHAR(50),
    z_score DECIMAL(10, 2),
    recommendation TEXT
) AS $$
DECLARE
    v_avg_amount DECIMAL(15, 2);
    v_std_amount DECIMAL(15, 2);
    v_expected_max DECIMAL(15, 2);
    v_z_score DECIMAL(10, 2);
BEGIN
    -- Get customer statistics
    SELECT 
        AVG(amount),
        STDDEV(amount)
    INTO v_avg_amount, v_std_amount
    FROM transactions
    WHERE customer_id = p_customer_id;
    
    -- Get expected max from profile
    SELECT expected_max_amount
    INTO v_expected_max
    FROM customer_profiles
    WHERE customer_id = p_customer_id;
    
    -- Calculate z-score
    IF v_std_amount > 0 THEN
        v_z_score := ABS((p_amount - v_avg_amount) / v_std_amount);
    ELSE
        v_z_score := 0;
    END IF;
    
    -- Determine anomaly
    IF p_amount > v_expected_max * 2 OR v_z_score > 4 THEN
        RETURN QUERY SELECT 
            TRUE,
            'CRITICAL_AMOUNT_ANOMALY'::VARCHAR(50),
            v_z_score,
            'BLOCK_TRANSACTION_IMMEDIATE_REVIEW'::TEXT;
    ELSIF p_amount > v_expected_max OR v_z_score > 3 THEN
        RETURN QUERY SELECT 
            TRUE,
            'AMOUNT_ANOMALY'::VARCHAR(50),
            v_z_score,
            'MANUAL_REVIEW_REQUIRED'::TEXT;
    ELSE
        RETURN QUERY SELECT 
            FALSE,
            'NORMAL'::VARCHAR(50),
            v_z_score,
            'APPROVE_TRANSACTION'::TEXT;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE ON customer_profiles TO application_user;
-- GRANT SELECT, INSERT, UPDATE ON transaction_alerts TO application_user;
-- GRANT SELECT, INSERT ON document_evidence TO application_user;
-- GRANT SELECT, INSERT, UPDATE ON compliance_reports TO application_user;
-- GRANT SELECT ON customer_risk_dashboard TO application_user;

COMMENT ON TABLE customer_profiles IS 'Customer business profiles extracted from KYC documents using LLM';
COMMENT ON TABLE transaction_alerts IS 'Real-time alerts from unified transaction + document monitoring';
COMMENT ON TABLE document_evidence IS 'Links between documents and transaction alerts (RAG system results)';
COMMENT ON TABLE compliance_reports IS 'Automated regulatory reports combining transaction + document analysis';
COMMENT ON VIEW customer_risk_dashboard IS 'Unified view of customer risk across all data sources';
COMMENT ON FUNCTION check_transaction_deviation IS 'Real-time transaction validation against customer profile';
