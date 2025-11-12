# Contributing Guide

This guide explains how to set up, develop, and contribute to the Banking Platform Data Migration Engine.

## Quick Start

### Prerequisites
- Java 21 LTS
- Maven 3.9+
- PostgreSQL 15
- Docker (for integration tests)
- Git

### Initial Setup

1. **Clone the repository**
```bash
git clone https://github.com/HermanQin9/fraud_test.git
cd fraud_test
```

2. **Build the project**
```bash
mvn clean install
```

3. **Run tests**
```bash
mvn test
```

4. **Start local database** (Docker)
```bash
docker-compose up -d
```

---

## Development Workflow

### Creating a Branch
```bash
git checkout -b feature/your-feature-name
```

### Making Changes
1. Write code following existing patterns
2. Add unit tests for new functionality
3. Ensure all tests pass: `mvn test`
4. Format code consistently

### Committing Changes
Use conventional commit messages:
```bash
git commit -m "feat: Add new data reader for XML format"
git commit -m "fix: Resolve connection pool timeout issue"
git commit -m "docs: Update README with new examples"
git commit -m "test: Add unit tests for normalizer"
git commit -m "refactor: Optimize batch insert logic"
```

**Commit types**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Testing
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Build/tooling

### Pushing Changes
```bash
git push origin feature/your-feature-name
```

---

## Code Standards

### Java Code
- **Language**: All code and comments in English
- **Style**: Follow standard Java conventions
- **Documentation**: JavaDoc for all public methods
- **Logging**: Use SLF4J (DEBUG, INFO, WARN, ERROR)
- **Testing**: Minimum 80% code coverage

### Example Code Structure
```java
package com.bankfraud.reader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads transaction data from XML files.
 */
public class XmlDataReader implements DataReader {
    private static final Logger logger = LoggerFactory.getLogger(XmlDataReader.class);
    
    @Override
    public List<Map<String, String>> read(String filePath) throws IOException {
        logger.info("Reading XML file: {}", filePath);
        // Implementation
    }
}
```

### Scala Code
- **Style**: Functional programming preferred
- **Immutability**: Use immutable data structures
- **Documentation**: ScalaDoc for public APIs
- **Testing**: ScalaTest for all Scala modules

---

## Testing Requirements

### Unit Tests
- Test all public methods
- Use JUnit 5 for Java
- Use ScalaTest for Scala
- Mock external dependencies with Mockito

### Integration Tests
- Use Testcontainers for database tests
- Test end-to-end workflows
- Clean up test data after execution

### Running Tests
```bash
# All tests
mvn test

# Specific test class
mvn test -Dtest=YourTestClass

# With coverage
mvn test jacoco:report
```

---

## Database Migrations

### Creating Migrations
Place Flyway SQL scripts in `src/main/resources/db/migration/`:
```
V1__Create_tables.sql
V2__Add_indexes.sql
V3__Update_schema.sql
```

### Running Migrations
```bash
mvn flyway:migrate
```

---

## Documentation

### Required Documentation
- **JavaDoc**: All public classes and methods
- **README updates**: For new features
- **Test documentation**: For complex test scenarios
- **Code comments**: For non-obvious logic

### Documentation Style
```java
/**
 * Normalizes transaction data from multiple source formats.
 * 
 * @param rawData Raw transaction data as key-value map
 * @param sourceFormat Format identifier (CSV, JSON, FIXED_WIDTH)
 * @return Normalized Transaction object
 * @throws IllegalArgumentException if required fields are missing
 */
public Transaction normalize(Map<String, String> rawData, String sourceFormat) {
    // Implementation
}
```

---

## Pull Request Process

1. **Ensure tests pass**
```bash
mvn clean test
```

2. **Update documentation**
- Update README.md if adding features
- Add JavaDoc/ScalaDoc
- Update TESTING.md if adding tests

3. **Create pull request**
- Provide clear description
- Reference any related issues
- Include test results

4. **Code review**
- Address reviewer comments
- Keep commits clean and focused

---

## Project Structure

```
BankFraudTest/
├── src/
│   ├── main/
│   │   ├── java/com/bankfraud/     # Java source code
│   │   ├── scala/com/bankfraud/    # Scala source code
│   │   ├── resources/              # Configuration files
│   │   └── scripts/                # Shell scripts
│   └── test/                       # All test code
├── docs/                           # Documentation
├── data/                           # Sample datasets
├── docker/                         # Docker configurations
└── pom.xml                         # Maven configuration
```

---

## Getting Help

### Resources
- **README.md**: Project overview and setup
- **docs/TESTING.md**: Testing documentation
- **docs/SCALA_MODULE.md**: Scala implementation guide
- **Source code**: Well-documented with comments

### Questions
- Check existing documentation first
- Review similar existing code
- Open an issue for discussion

---

## Environment Variables

### Database Configuration
```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=frauddb
export PGUSER=postgres
export PGPASSWORD=postgres
```

### AWS Configuration (Optional)
```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

---

## Troubleshooting

### Build Fails
```bash
# Clean and rebuild
mvn clean install -U
```

### Tests Fail
```bash
# Check Docker is running
docker ps

# Verify database connection
psql -h localhost -U postgres -d frauddb
```

### Connection Issues
- Ensure PostgreSQL is running
- Check connection pool settings in `application.properties`
- Verify firewall allows connections

---

## License

This project is licensed under the MIT License - see LICENSE file for details.

---

## Contact

- **Repository**: https://github.com/HermanQin9/fraud_test
- **Issues**: Open an issue on GitHub
- **Owner**: HermanQin9

---

**Last Updated**: November 6, 2025
