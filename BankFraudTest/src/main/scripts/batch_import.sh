#!/bin/bash

###############################################################################
# Batch Import Script
# Imports transaction data into PostgreSQL database
# 
# Author: Banking Platform Team
# Version: 1.0
###############################################################################

set -e  # Exit on error

# Configuration
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-21}"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./logs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/batch_import_${TIMESTAMP}.log"

# Database configuration (override with environment variables)
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

# Check prerequisites
check_prerequisites() {
    log "INFO" "Checking prerequisites"
    
    # Check Java installation
    if [ ! -d "${JAVA_HOME}" ]; then
        log "ERROR" "JAVA_HOME not found: ${JAVA_HOME}"
        exit 1
    fi
    
    # Check if Maven is available
    if ! command -v mvn &> /dev/null; then
        log "ERROR" "Maven not found. Please install Maven."
        exit 1
    fi
    
    # Check database connectivity
    if command -v psql &> /dev/null; then
        if ! PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -c "SELECT 1" &> /dev/null; then
            log "WARN" "Cannot connect to database. Please verify database is running."
        else
            log "INFO" "Database connection successful"
        fi
    else
        log "WARN" "psql not found, skipping database connectivity check"
    fi
    
    log "INFO" "Prerequisites check complete"
}

# Build project
build_project() {
    log "INFO" "Building project with Maven"
    cd "${PROJECT_DIR}"
    
    if mvn clean package -DskipTests >> "${LOG_FILE}" 2>&1; then
        log "INFO" "Project build successful"
    else
        log "ERROR" "Project build failed. Check log: ${LOG_FILE}"
        exit 1
    fi
}

# Import single file
import_file() {
    local file=$1
    local format=$2
    
    log "INFO" "Importing file: ${file} (format: ${format})"
    
    # Run Java application with file path and format
    java -cp "${PROJECT_DIR}/target/banking-platform-migration-1.0.0.jar:${PROJECT_DIR}/target/lib/*" \
         com.bankfraud.service.DataIngestionService \
         "${file}" "${format}" >> "${LOG_FILE}" 2>&1
    
    if [ $? -eq 0 ]; then
        log "INFO" "Successfully imported: ${file}"
        return 0
    else
        log "ERROR" "Failed to import: ${file}"
        return 1
    fi
}

# Batch import all files
batch_import() {
    log "INFO" "Starting batch import from: ${DATA_DIR}"
    
    local success_count=0
    local failure_count=0
    
    # Import CSV files
    for file in $(find "${DATA_DIR}" -name "*.csv" -type f); do
        if import_file "${file}" "CSV"; then
            success_count=$((success_count + 1))
        else
            failure_count=$((failure_count + 1))
        fi
    done
    
    # Import JSON files
    for file in $(find "${DATA_DIR}" -name "*.json" -type f); do
        if import_file "${file}" "JSON"; then
            success_count=$((success_count + 1))
        else
            failure_count=$((failure_count + 1))
        fi
    done
    
    # Import fixed-width files
    for file in $(find "${DATA_DIR}" -name "*.txt" -type f); do
        if import_file "${file}" "FIXED_WIDTH"; then
            success_count=$((success_count + 1))
        else
            failure_count=$((failure_count + 1))
        fi
    done
    
    log "INFO" "Batch import complete: Success=${success_count}, Failures=${failure_count}"
}

# Generate import summary
generate_summary() {
    log "INFO" "Generating import summary"
    
    if command -v psql &> /dev/null; then
        local total_transactions=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM transactions" 2>/dev/null || echo "N/A")
        local fraud_transactions=$(PGPASSWORD="${DB_PASSWORD}" psql -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" -d "${DB_NAME}" -t -c "SELECT COUNT(*) FROM transactions WHERE fraud_flag = true" 2>/dev/null || echo "N/A")
        
        log "INFO" "Database Summary:"
        log "INFO" "  Total Transactions: ${total_transactions}"
        log "INFO" "  Fraudulent Transactions: ${fraud_transactions}"
    fi
}

# Main workflow
main() {
    log "INFO" "========================================="
    log "INFO" "Batch Import Process Started"
    log "INFO" "========================================="
    
    check_prerequisites
    build_project
    batch_import
    generate_summary
    
    log "INFO" "========================================="
    log "INFO" "Batch Import Process Complete"
    log "INFO" "Log file: ${LOG_FILE}"
    log "INFO" "========================================="
}

# Execute main function
main "$@"
