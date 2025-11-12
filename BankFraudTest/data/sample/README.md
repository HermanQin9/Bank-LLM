# Sample Banking Transaction Datasets

This directory contains sample transaction data from three different banking platforms, demonstrating multi-source data migration scenarios.

## Dataset Overview

### 1. **bank_a_transactions.csv**
**Source**: Legacy Bank A System 
**Format**: CSV (Comma-Separated Values) 
**Records**: 30 transactions 
**Schema**:
```
transaction_id,customer_id,transaction_date,amount,currency,
merchant_name,merchant_category,card_last_four,location_country,
location_city,transaction_type
```

**Characteristics**:
- Standard CSV format with headers
- Traditional retail and e-commerce transactions
- ISO 8601 datetime format
- US-based transactions

---

### 2. **bank_b_transactions.json**
**Source**: Modern Bank B API 
**Format**: JSON (JavaScript Object Notation) 
**Records**: 15 transactions 
**Schema**:
```json
{
 "txnId": string,
 "accountId": string,
 "timestamp": ISO 8601 timestamp,
 "transactionAmount": {
 "value": number,
 "currencyCode": string
 },
 "merchant": {
 "name": string,
 "category": string,
 "mcc": string
 },
 "cardInfo": {
 "last4": string,
 "type": string
 },
 "location": {
 "country": string,
 "city": string,
 "coordinates": { "lat": number, "lon": number }
 },
 "channel": string,
 "status": string
}
```

**Characteristics**:
- Nested JSON structure
- Digital/online payment focus (Stripe, PayPal, AWS, etc.)
- Includes geolocation coordinates
- MCC (Merchant Category Code) included
- Multiple payment channels (Online, Mobile, Recurring)

---

### 3. **bank_c_fixed_width.txt**
**Source**: Mainframe Legacy Bank C System 
**Format**: Fixed-width text file (No delimiters) 
**Records**: 30 transactions 
**Field Positions**:
```
Position Length Field Name
1-8 8 Transaction ID
9-17 9 Customer ID
18-31 14 Transaction Date (YYYYMMDDHHMMSS)
32-39 8 Amount (in cents)
40-42 3 Currency
43-62 20 Merchant Name (padded)
63-72 10 Merchant Category (padded)
73-76 4 MCC Code
77-78 2 Country Code
79-92 14 City (padded)
```

**Characteristics**:
- No delimiters - positions matter!
- Mainframe-style format
- Fixed-width fields with space padding
- Compact date representation
- Amount stored in cents

---

## Data Migration Challenges

Each format presents unique challenges that this project addresses:

### Format Differences
| Aspect | Bank A (CSV) | Bank B (JSON) | Bank C (Fixed) |
|--------|-------------|---------------|----------------|
| **Delimiter** | Comma | N/A (JSON) | None (position-based) |
| **Date Format** | `YYYY-MM-DD HH:MM:SS` | ISO 8601 with TZ | `YYYYMMDDHHMMSS` |
| **Amount** | Decimal | Nested object | Cents (integer) |
| **Structure** | Flat | Nested | Flat/Packed |
| **Headers** | Yes | Key-value | No |
| **Parsing** | Easy | Medium | Hard |

### Normalization Requirements
1. **ID Mapping**: Different ID formats across systems
2. **Date Standardization**: Convert all to PostgreSQL TIMESTAMP
3. **Amount Conversion**: Handle cents vs. decimal, nested values
4. **String Trimming**: Remove padding from fixed-width fields
5. **Schema Mapping**: Flatten nested JSON structures
6. **Null Handling**: Different systems handle missing data differently

---

## Expected Normalized Output

All three formats should be transformed to this unified schema:

```sql
CREATE TABLE transactions (
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
 source_system VARCHAR(50), -- 'BANK_A', 'BANK_B', 'BANK_C'
 raw_data JSONB, -- Original data
 created_at TIMESTAMP DEFAULT NOW()
);
```

---

## Data Statistics

| Metric | Bank A | Bank B | Bank C | Total |
|--------|--------|--------|--------|-------|
| **Records** | 30 | 15 | 30 | **75** |
| **File Size** | ~2.5 KB | ~6 KB | ~3 KB | ~11.5 KB |
| **Avg Amount** | $1,245 | $3,126 | $358 | $1,576 |
| **Date Range** | Nov 1-3, 2024 | Nov 1-3, 2024 | Nov 1-3, 2024 | Nov 1-3, 2024 |
| **Unique Customers** | 15 | 9 | 13 | **37** |
| **Unique Merchants** | 30 | 15 | 30 | **75** |

---

## Testing Scenarios

These datasets support testing:

1. **Multi-Format Parsing**
 - CSV parser with comma handling
 - JSON parser with nested objects
 - Fixed-width parser with position-based extraction

2. **Data Validation**
 - Amount validation (positive values)
 - Date format validation
 - Required field checks
 - Currency code validation

3. **Data Quality**
 - Duplicate detection across sources
 - Data completeness checks
 - Format consistency validation

4. **Performance**
 - Batch insert optimization
 - Parallel processing (multiple files)
 - Error handling and recovery

5. **Traceability**
 - Source system tracking
 - Original data preservation (raw_data JSONB)
 - Import logging

---

## Usage Example

```java
// Example: Import all three formats
IngestionPipeline pipeline = new IngestionPipeline();

// Bank A (CSV)
ImportResult resultA = pipeline.process(
 "data/sample/bank_a_transactions.csv", 
 DataSource.BANK_A
);

// Bank B (JSON)
ImportResult resultB = pipeline.process(
 "data/sample/bank_b_transactions.json", 
 DataSource.BANK_B
);

// Bank C (Fixed-width)
ImportResult resultC = pipeline.process(
 "data/sample/bank_c_fixed_width.txt", 
 DataSource.BANK_C
);

System.out.println("Total imported: " + 
 (resultA.getSuccessCount() + 
 resultB.getSuccessCount() + 
 resultC.getSuccessCount()) + " records");
```

---

## Data Generation Notes

- **Synthetic Data**: All data is artificially generated for demonstration
- **No PII**: No real personally identifiable information
- **Realistic**: Mimics real banking transaction patterns
- **Diverse**: Covers various merchant types and transaction amounts
- **Safe for GitHub**: Can be publicly shared

---

## Additional Data Sources

For larger datasets and real-world testing:

1. **Kaggle Credit Card Transactions** 
 https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions

2. **PaySim Mobile Money Dataset** 
 https://www.kaggle.com/datasets/ealaxi/paysim1

3. **Berka Czech Banking Dataset** 
 https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions

---

**Generated**: 2024-11-05 
**Purpose**: Multi-source banking data migration demonstration 
**License**: Public Domain (Synthetic Data)
