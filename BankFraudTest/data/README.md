# Data Directory

This directory contains real-world banking and financial transaction datasets used for demonstrating multi-source data migration and normalization capabilities.

## Directory Structure

```
data/
 sample/ # Small sample datasets (3 formats)
 bank_a_transactions.csv # CSV format (30 records)
 bank_b_transactions.json # JSON format (15 records)
 bank_c_fixed_width.txt # Fixed-width format (30 records)
 README.md # Sample data documentation

 credit_card/ # Real credit card transactions
 User0_credit_card_transactions.csv (1.81 MB, ~24K transactions)

 german_credit/ # German credit approval data
 [German credit data files]

 lending_club/ # (To be extracted)
 accepted_2007_to_2018Q4.csv.gz (374 MB, 2.2M+ loans)
```

---

## Available Datasets

### 1. **Sample Multi-Format Data** (Generated)
**Location**: `sample/` 
**Purpose**: Demonstrate multi-source data ingestion from different banking systems 
**Formats**: CSV, JSON, Fixed-width text 
**Total Records**: 75 transactions 
**Use Case**: Quick testing and development

**Key Features**:
- 3 different data formats
- Simulates legacy and modern banking systems
- Small size (< 1 MB total)
- Safe for GitHub

---

### 2. **Credit Card Transactions** Real Data
**Location**: `credit_card/User0_credit_card_transactions.csv` 
**Source**: Real anonymized credit card transactions 
**Size**: 1.81 MB 
**Records**: ~24,000 transactions 
**Time Period**: 2002-2005 

**Schema**:
```
User, Card, Year, Month, Day, Time, Amount, Use Chip,
Merchant Name, Merchant City, Merchant State, Zip, MCC,
Errors?, Is Fraud?
```

**Fields**:
- User/Card ID
- Transaction datetime (Year, Month, Day, Time)
- Amount (USD)
- Payment method (Chip/Swipe)
- Merchant information (Name, City, State, Zip)
- MCC (Merchant Category Code)
- Fraud indicator

**Use Cases**:
- Data normalization practice
- Date/time parsing and standardization
- Amount formatting
- Merchant categorization
- Fraud pattern analysis

---

### 3. **Lending Club Loan Data** Real Data (Large)
**Location**: `accepted_2007_to_2018Q4.csv.gz` 
**Source**: Lending Club - Real P2P lending platform data 
**Size**: 374 MB (compressed), ~1.6 GB (uncompressed) 
**Records**: 2.2M+ loans 
**Time Period**: 2007-2018 

**Schema**: 150+ columns including:
- Loan amount, interest rate, term
- Borrower information (anonymized)
- Credit score, income
- Loan status, default rate
- Payment history

**Use Cases**:
- Large-scale data processing
- Batch import optimization
- Data quality validation
- Performance benchmarking

**Note**: Large file - extract on demand 
```bash
# To extract (PowerShell):
Expand-Archive -Path accepted_2007_to_2018Q4.csv.gz -DestinationPath lending_club/
```

---

### 4. **German Credit Data** Real Data (Small)
**Location**: `german_credit/` 
**Source**: UCI Machine Learning Repository - Real credit approval data 
**Size**: < 100 KB 
**Records**: 1,000 credit applications 

**Schema**: 20 attributes including:
- Credit history
- Purpose of credit
- Credit amount
- Employment status
- Personal status
- Property ownership
- Credit risk classification

**Use Cases**:
- Data transformation practice
- Feature engineering
- Classification modeling
- Small dataset testing

---

## Data Migration Strategy

### Primary Dataset for Demo: Credit Card Transactions
**Why**: 
- Real data (most authentic)
- Manageable size (1.81 MB)
- Rich schema (15 fields)
- Good for ETL demonstration
- Can upload to GitHub

### Conversion Plan:
Transform `User0_credit_card_transactions.csv` into 3 formats:

1. **CSV (Bank A format)** - Keep original structure
2. **JSON (Bank B format)** - Nested structure with merchant object
3. **Fixed-width (Bank C format)** - Mainframe-style format

This demonstrates:
- Multi-format data ingestion
- Data normalization across systems
- Schema standardization
- Real-world data handling

---

## Data Preparation Tasks

### Phase 1: Extract and Clean
- [x] Download datasets
- [x] Extract compressed files
- [ ] Create unified schema
- [ ] Sample data for testing

### Phase 2: Format Conversion
- [ ] Convert CSV to JSON format
- [ ] Convert CSV to fixed-width format
- [ ] Validate data integrity
- [ ] Document schema mappings

### Phase 3: Integration
- [ ] Load data into PostgreSQL
- [ ] Create import scripts
- [ ] Add data validation
- [ ] Performance testing

---

## Data Privacy & Usage

### Credit Card Data
- Anonymized (no real PII)
- Publicly available dataset
- Safe for GitHub and demonstration
- Do not use for actual fraud prevention

### Lending Club Data
- Official public release
- Anonymized borrower information
- Can be used for research/education
- Large file - consider .gitignore

### German Credit Data
- UCI ML Repository (public)
- Academic research standard
- No privacy concerns

---

## Data Statistics Summary

| Dataset | Records | Size | Format | Real/Synthetic |
|---------|---------|------|--------|---------------|
| Sample (3 files) | 75 | < 1 MB | Multi | Synthetic |
| Credit Card | 24K | 1.81 MB | CSV | Real |
| Lending Club | 2.2M | 374 MB | CSV.gz | Real |
| German Credit | 1K | < 0.1 MB | CSV | Real |
| **Total** | **2.2M+** | **~376 MB** | - | **Mostly Real** |

---

## Quick Start

### Load Sample Data (Quick Test)
```bash
cd data/sample
# Use small sample datasets for rapid development
```

### Load Real Data (Demo)
```bash
cd data/credit_card
# Use User0_credit_card_transactions.csv for realistic demo
```

### Performance Testing (Optional)
```bash
cd data
# Extract lending_club data for large-scale testing
gunzip -k accepted_2007_to_2018Q4.csv.gz
```

---

## References

- **Credit Card Data**: Various public sources
- **Lending Club**: https://www.lendingclub.com/
- **German Credit**: https://archive.ics.uci.edu/ml/
- **Sample Data**: Project-generated for demonstration

---

**Last Updated**: November 5, 2025 
**Purpose**: Banking platform data migration demonstration 
**Project**: Enterprise Banking Data Migration Engine
