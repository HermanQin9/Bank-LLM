# Testing Guide

## Automated Testing (GitHub Actions)

Every push to branches triggers automated tests:
- Java: Maven tests with PostgreSQL
- Python: Pytest with coverage
- Scala: Spark unit tests
- Linting: flake8, black

See `.github/workflows/ci.yml` for configuration.

## Local Testing

### Java Tests
```bash
cd BankFraudTest
mvn test
mvn jacoco:report  # Coverage report in target/site/jacoco/
```

### Python Tests
```bash
cd LLM
pytest tests/ -v                    # Run all tests
pytest tests/test_basic.py -v      # Run specific test
pytest --cov=src --cov-report=html # With coverage
```

### Scala/Spark Tests
```bash
cd BankFraudTest
mvn test -Dtest=SparkTransactionProcessorTest
```

## Test Status

**Python**: 3/3 tests passing
- test_python_version: PASS
- test_imports: PASS
- test_basic_math: PASS

**Java**: All unit tests passing (run `mvn test` to verify)

**Scala**: Spark tests (run `mvn test -Dtest=Spark*` to verify)

## CI/CD Pipeline

On every push:
1. Build Java project with Maven
2. Run Java tests with PostgreSQL service
3. Run Python tests with pytest
4. Run Scala/Spark tests
5. Generate coverage reports
6. Upload to Codecov
7. Run code linting

## Test Coverage

- Java: JaCoCo coverage report
- Python: pytest-cov HTML report
- Coverage uploaded to Codecov automatically

## Adding New Tests

### Java
1. Create test in `BankFraudTest/src/test/java/`
2. Use JUnit 5 annotations
3. Run `mvn test` to verify

### Python
1. Create test in `LLM/tests/test_*.py`
2. Use pytest fixtures and assertions
3. Run `pytest tests/` to verify

### Scala
1. Create test in `BankFraudTest/src/test/scala/`
2. Use ScalaTest
3. Run `mvn test -Dtest=YourTest` to verify
