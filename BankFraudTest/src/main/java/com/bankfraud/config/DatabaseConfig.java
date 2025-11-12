package com.bankfraud.config;

import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sql.DataSource;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Database configuration manager using HikariCP connection pooling.
 * Loads configuration from application.properties.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class DatabaseConfig {

    private static final Logger logger = LoggerFactory.getLogger(DatabaseConfig.class);
    private static final String PROPERTIES_FILE = "application.properties";

    private static HikariDataSource dataSource;

    /**
     * Gets the singleton DataSource instance.
     * Initializes on first call with configuration from application.properties.
     * 
     * @return HikariCP DataSource
     */
    public static synchronized DataSource getDataSource() {
        if (dataSource == null) {
            logger.info("Initializing database connection pool");
            dataSource = createDataSource();
        }
        return dataSource;
    }

    /**
     * Creates and configures HikariCP DataSource.
     * 
     * @return configured HikariDataSource
     */
    private static HikariDataSource createDataSource() {
        Properties properties = loadProperties();

        HikariConfig config = new HikariConfig();

        // Database connection properties - check System properties first (for tests)
        String jdbcUrl = System.getProperty("db.url",
                properties.getProperty("db.url", "jdbc:postgresql://localhost:5432/bankfraud"));
        String username = System.getProperty("db.username",
                properties.getProperty("db.username", "postgres"));
        String password = System.getProperty("db.password",
                properties.getProperty("db.password", ""));

        config.setJdbcUrl(jdbcUrl);
        config.setUsername(username);
        config.setPassword(password);
        config.setDriverClassName(properties.getProperty("db.driver", "org.postgresql.Driver"));

        // Connection pool configuration
        config.setMaximumPoolSize(Integer.parseInt(properties.getProperty("db.pool.max-size", "10")));
        config.setMinimumIdle(Integer.parseInt(properties.getProperty("db.pool.min-idle", "2")));
        config.setConnectionTimeout(Long.parseLong(properties.getProperty("db.pool.connection-timeout", "30000")));
        config.setIdleTimeout(Long.parseLong(properties.getProperty("db.pool.idle-timeout", "600000")));
        config.setMaxLifetime(Long.parseLong(properties.getProperty("db.pool.max-lifetime", "1800000")));

        // Connection testing
        config.setConnectionTestQuery("SELECT 1");

        // Pool name for monitoring
        config.setPoolName("BankFraudHikariPool");

        // Additional HikariCP optimizations
        config.addDataSourceProperty("cachePrepStmts", "true");
        config.addDataSourceProperty("prepStmtCacheSize", "250");
        config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");

        logger.info("Database connection pool configured: URL={}, MaxPoolSize={}, MinIdle={}",
                jdbcUrl, config.getMaximumPoolSize(), config.getMinimumIdle());

        return new HikariDataSource(config);
    }

    /**
     * Loads database properties from application.properties file.
     * 
     * @return Properties object
     */
    private static Properties loadProperties() {
        Properties properties = new Properties();

        try (InputStream input = DatabaseConfig.class.getClassLoader()
                .getResourceAsStream(PROPERTIES_FILE)) {

            if (input == null) {
                logger.warn("Unable to find {}, using default values", PROPERTIES_FILE);
                return properties;
            }

            properties.load(input);
            logger.debug("Successfully loaded database properties from {}", PROPERTIES_FILE);

        } catch (IOException e) {
            logger.error("Failed to load properties file: {}", PROPERTIES_FILE, e);
        }

        return properties;
    }

    /**
     * Closes the datasource and releases all connections.
     * Should be called during application shutdown.
     */
    public static synchronized void closeDataSource() {
        if (dataSource != null && !dataSource.isClosed()) {
            logger.info("Closing database connection pool");
            dataSource.close();
            dataSource = null;
        }
    }

    /**
     * Tests database connectivity.
     * 
     * @return true if connection is successful, false otherwise
     */
    public static boolean testConnection() {
        try (var connection = getDataSource().getConnection()) {
            boolean isValid = connection.isValid(5);
            logger.info("Database connection test: {}", isValid ? "SUCCESS" : "FAILED");
            return isValid;
        } catch (Exception e) {
            logger.error("Database connection test failed: {}", e.getMessage(), e);
            return false;
        }
    }
}
