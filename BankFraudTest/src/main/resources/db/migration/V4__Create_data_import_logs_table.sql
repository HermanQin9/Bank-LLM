-- V4: Create data_import_logs table
-- This table tracks ETL job execution and performance

CREATE TABLE IF NOT EXISTS data_import_logs (
    log_id SERIAL PRIMARY KEY,
    source_system VARCHAR(50) NOT NULL,
    file_name VARCHAR(255),
    s3_key VARCHAR(500),
    records_processed INTEGER DEFAULT 0,
    records_success INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    error_message TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'IN_PROGRESS',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_import_logs_source_system ON data_import_logs(source_system);
CREATE INDEX idx_import_logs_status ON data_import_logs(status);
CREATE INDEX idx_import_logs_start_time ON data_import_logs(start_time DESC);
CREATE INDEX idx_import_logs_created_at ON data_import_logs(created_at DESC);

-- Check constraint
ALTER TABLE data_import_logs ADD CONSTRAINT chk_import_status 
    CHECK (status IN ('IN_PROGRESS', 'COMPLETED', 'FAILED', 'PARTIAL'));

-- Comments
COMMENT ON TABLE data_import_logs IS 'ETL job execution logs and performance metrics';
COMMENT ON COLUMN data_import_logs.s3_key IS 'S3 object key if data source is from AWS S3';
COMMENT ON COLUMN data_import_logs.duration_seconds IS 'Total execution time in seconds';
COMMENT ON COLUMN data_import_logs.status IS 'Job status: IN_PROGRESS, COMPLETED, FAILED, PARTIAL';
