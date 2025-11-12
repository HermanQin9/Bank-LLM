#!/bin/bash

###############################################################################
# Data Preprocessing Script
# Validates and cleans input data files before ingestion
# 
# Author: Banking Platform Team
# Version: 1.0
###############################################################################

set -e  # Exit on error

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
LOG_DIR="${LOG_DIR:-./logs}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/preprocess_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Logging function
log() {
    local level=$1
    shift
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [${level}] $*" | tee -a "${LOG_FILE}"
}

# Validate CSV file
validate_csv() {
    local file=$1
    log "INFO" "Validating CSV file: ${file}"
    
    if [ ! -f "${file}" ]; then
        log "ERROR" "File does not exist: ${file}"
        return 1
    fi
    
    # Check if file is empty
    if [ ! -s "${file}" ]; then
        log "ERROR" "File is empty: ${file}"
        return 1
    fi
    
    # Check if file has headers
    local header_count=$(head -n 1 "${file}" | tr ',' '\n' | wc -l)
    if [ "${header_count}" -lt 2 ]; then
        log "ERROR" "File has insufficient columns: ${file}"
        return 1
    fi
    
    # Count records (excluding header)
    local record_count=$(($(wc -l < "${file}") - 1))
    log "INFO" "File validated: ${file} - ${record_count} records"
    
    return 0
}

# Clean CSV file (remove empty lines, trim whitespace)
clean_csv() {
    local input_file=$1
    local output_file=$2
    
    log "INFO" "Cleaning CSV file: ${input_file} -> ${output_file}"
    
    # Remove empty lines and trailing whitespace
    grep -v '^[[:space:]]*$' "${input_file}" | sed 's/[[:space:]]*$//' > "${output_file}"
    
    local cleaned_count=$(wc -l < "${output_file}")
    log "INFO" "Cleaned file created: ${output_file} - ${cleaned_count} lines"
}

# Validate JSON file
validate_json() {
    local file=$1
    log "INFO" "Validating JSON file: ${file}"
    
    if [ ! -f "${file}" ]; then
        log "ERROR" "File does not exist: ${file}"
        return 1
    fi
    
    # Check JSON syntax using python
    if command -v python3 &> /dev/null; then
        if ! python3 -m json.tool "${file}" > /dev/null 2>&1; then
            log "ERROR" "Invalid JSON syntax: ${file}"
            return 1
        fi
    else
        log "WARN" "Python3 not found, skipping JSON syntax validation"
    fi
    
    log "INFO" "File validated: ${file}"
    return 0
}

# Main preprocessing workflow
main() {
    log "INFO" "Starting data preprocessing"
    log "INFO" "Data directory: ${DATA_DIR}"
    
    # Process all CSV files in data directory
    local csv_files=$(find "${DATA_DIR}" -name "*.csv" -type f)
    local csv_count=0
    
    for file in ${csv_files}; do
        if validate_csv "${file}"; then
            # Create cleaned version in temp directory
            local basename=$(basename "${file}")
            local cleaned_file="${DATA_DIR}/.cleaned_${basename}"
            clean_csv "${file}" "${cleaned_file}"
            csv_count=$((csv_count + 1))
        fi
    done
    
    # Process all JSON files
    local json_files=$(find "${DATA_DIR}" -name "*.json" -type f)
    local json_count=0
    
    for file in ${json_files}; do
        if validate_json "${file}"; then
            json_count=$((json_count + 1))
        fi
    done
    
    log "INFO" "Preprocessing complete: ${csv_count} CSV files, ${json_count} JSON files"
    log "INFO" "Log file: ${LOG_FILE}"
}

# Execute main function
main "$@"
