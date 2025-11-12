#!/bin/bash

###############################################################################
# Data Validation Script
# Validates imported data in PostgreSQL database
# 
# Author: Banking Platform Team
# Version: 1.0
###############################################################################

set -e  # Exit on error

# Configuration
LOG_DIR="${LOG_DIR:-./logs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/validation_${TIMESTAMP}.log"

# Database configuration
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-bankfraud}"
DB_USER="${DB_USER:-postgres}"

# Create log directory
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "${LOG_FILE}"
}

# Execute SQL query
execute_query() {
    local query=$1
    PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "${query}" 2>/dev/null
}

# Check database connectivity
check_database() {
    log "INFO" "Checking database connectivity"
    
    if ! command -v psql &> /dev/null; then
        log "ERROR" "psql not found. Please install PostgreSQL client."
        exit 1
    fi
    
    if ! execute_query "SELECT 1" &> /dev/null; then
        log "ERROR" "Cannot connect to database: ${DB_NAME}@${DB_HOST}:${DB_PORT}"
        exit 1
    fi
    
    log "INFO" "Database connection successful"
}

# Validate data completeness
validate_completeness() {
    log "INFO" "Validating data completeness"
    
    local total_count=$(execute_query "SELECT COUNT(*) FROM transactions")
    log "INFO" "Total transactions: ${total_count}"
    
    local null_amount=$(execute_query "SELECT COUNT(*) FROM transactions WHERE amount IS NULL")
    log "INFO" "Transactions with null amount: ${null_amount}"
    
    local null_customer=$(execute_query "SELECT COUNT(*) FROM transactions WHERE customer_id IS NULL")
    log "INFO" "Transactions with null customer_id: ${null_customer}"
    
    local null_date=$(execute_query "SELECT COUNT(*) FROM transactions WHERE transaction_date IS NULL")
    log "INFO" "Transactions with null date: ${null_date}"
    
    if [ "${null_amount}" -gt 0 ] || [ "${null_date}" -gt 0 ]; then
        log "WARN" "Data completeness issues detected"
        return 1
    fi
    
    log "INFO" "Data completeness validation passed"
    return 0
}

# Validate data integrity
validate_integrity() {
    log "INFO" "Validating data integrity"
    
    # Check for duplicate transaction IDs
    local duplicate_ids=$(execute_query "SELECT COUNT(*) FROM (SELECT transaction_id, COUNT(*) as cnt FROM transactions GROUP BY transaction_id HAVING COUNT(*) > 1) as dupes")
    log "INFO" "Duplicate transaction IDs: ${duplicate_ids}"
    
    # Check for negative amounts
    local negative_amounts=$(execute_query "SELECT COUNT(*) FROM transactions WHERE amount < 0")
    log "INFO" "Transactions with negative amounts: ${negative_amounts}"
    
    # Check for future dates
    local future_dates=$(execute_query "SELECT COUNT(*) FROM transactions WHERE transaction_date > NOW()")
    log "INFO" "Transactions with future dates: ${future_dates}"
    
    if [ "${duplicate_ids}" -gt 0 ] || [ "${negative_amounts}" -gt 0 ] || [ "${future_dates}" -gt 0 ]; then
        log "WARN" "Data integrity issues detected"
        return 1
    fi
    
    log "INFO" "Data integrity validation passed"
    return 0
}

# Validate data quality
validate_quality() {
    log "INFO" "Validating data quality"
    
    # Check average transaction amount
    local avg_amount=$(execute_query "SELECT ROUND(AVG(amount)::numeric, 2) FROM transactions")
    log "INFO" "Average transaction amount: \$${avg_amount}"
    
    # Check date range
    local min_date=$(execute_query "SELECT MIN(transaction_date)::date FROM transactions")
    local max_date=$(execute_query "SELECT MAX(transaction_date)::date FROM transactions")
    log "INFO" "Date range: ${min_date} to ${max_date}"
    
    # Check fraud rate
    local total=$(execute_query "SELECT COUNT(*) FROM transactions")
    local fraud=$(execute_query "SELECT COUNT(*) FROM transactions WHERE fraud_flag = true")
    local fraud_rate=$(echo "scale=2; ${fraud} * 100 / ${total}" | bc)
    log "INFO" "Fraud rate: ${fraud_rate}%"
    
    # Check source system distribution
    log "INFO" "Source system distribution:"
    execute_query "SELECT source_system, COUNT(*) FROM transactions GROUP BY source_system" | while read line; do
        log "INFO" "  ${line}"
    done
    
    log "INFO" "Data quality validation complete"
    return 0
}

# Generate validation report
generate_report() {
    log "INFO" "Generating validation report"
    
    local report_file="${LOG_DIR}/validation_report_${TIMESTAMP}.txt"
    
    cat > "${report_file}" << EOF
========================================
Data Validation Report
========================================
Generated: $(date)
Database: ${DB_NAME}@${DB_HOST}:${DB_PORT}

SUMMARY
-------
Total Transactions: $(execute_query "SELECT COUNT(*) FROM transactions")
Fraudulent Transactions: $(execute_query "SELECT COUNT(*) FROM transactions WHERE fraud_flag = true")
Unique Customers: $(execute_query "SELECT COUNT(DISTINCT customer_id) FROM transactions WHERE customer_id IS NOT NULL")

DATE RANGE
----------
Earliest Transaction: $(execute_query "SELECT MIN(transaction_date)::date FROM transactions")
Latest Transaction: $(execute_query "SELECT MAX(transaction_date)::date FROM transactions")

AMOUNT STATISTICS
-----------------
Total Amount: \$$(execute_query "SELECT ROUND(SUM(amount)::numeric, 2) FROM transactions")
Average Amount: \$$(execute_query "SELECT ROUND(AVG(amount)::numeric, 2) FROM transactions")
Min Amount: \$$(execute_query "SELECT ROUND(MIN(amount)::numeric, 2) FROM transactions")
Max Amount: \$$(execute_query "SELECT ROUND(MAX(amount)::numeric, 2) FROM transactions")

SOURCE SYSTEM BREAKDOWN
-----------------------
$(execute_query "SELECT source_system, COUNT(*) as count, ROUND(AVG(amount)::numeric, 2) as avg_amount FROM transactions GROUP BY source_system")

TOP MERCHANTS
-------------
$(execute_query "SELECT merchant_name, COUNT(*) as count FROM transactions WHERE merchant_name IS NOT NULL GROUP BY merchant_name ORDER BY count DESC LIMIT 10")

VALIDATION STATUS
-----------------
See log file for details: ${LOG_FILE}

========================================
EOF
    
    log "INFO" "Validation report generated: ${report_file}"
}

# Main workflow
main() {
    log "INFO" "========================================="
    log "INFO" "Data Validation Process Started"
    log "INFO" "========================================="
    
    check_database
    
    local validation_status=0
    
    if ! validate_completeness; then
        validation_status=1
    fi
    
    if ! validate_integrity; then
        validation_status=1
    fi
    
    validate_quality
    generate_report
    
    if [ ${validation_status} -eq 0 ]; then
        log "INFO" "All validations passed"
    else
        log "WARN" "Some validations failed. Review log for details."
    fi
    
    log "INFO" "========================================="
    log "INFO" "Data Validation Process Complete"
    log "INFO" "Log file: ${LOG_FILE}"
    log "INFO" "========================================="
    
    exit ${validation_status}
}

# Execute main function
main "$@"
