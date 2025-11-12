package com.bankfraud.repository;

import com.bankfraud.config.DatabaseConfig;
import com.bankfraud.model.Customer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.*;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Repository for Customer entity database operations.
 * Handles CRUD operations and queries for customers.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class CustomerRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(CustomerRepository.class);
    
    private static final String INSERT_SQL = 
            "INSERT INTO customers (customer_id, first_name, last_name, email, phone, " +
            "account_created_date, risk_level, total_transactions, lifetime_value, created_at) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    
    private static final String SELECT_BY_ID_SQL = 
            "SELECT * FROM customers WHERE customer_id = ?";
    
    private static final String SELECT_ALL_SQL = 
            "SELECT * FROM customers ORDER BY created_at DESC";
    
    private static final String SELECT_BY_RISK_LEVEL_SQL = 
            "SELECT * FROM customers WHERE risk_level = ? ORDER BY lifetime_value DESC";
    
    private static final String UPDATE_SQL = 
            "UPDATE customers SET first_name = ?, last_name = ?, email = ?, phone = ?, " +
            "risk_level = ?, total_transactions = ?, lifetime_value = ?, last_transaction_date = ?, " +
            "updated_at = ? WHERE customer_id = ?";
    
    private static final String DELETE_SQL = 
            "DELETE FROM customers WHERE customer_id = ?";
    
    private static final String UPDATE_TRANSACTION_STATS_SQL = 
            "UPDATE customers SET total_transactions = total_transactions + 1, " +
            "lifetime_value = lifetime_value + ?, last_transaction_date = ?, updated_at = ? " +
            "WHERE customer_id = ?";
    
    /**
     * Inserts a new customer into the database.
     * 
     * @param customer the customer to insert
     * @return true if insert was successful
     */
    public boolean save(Customer customer) {
        logger.debug("Attempting to save customer: {}", customer.getCustomerId());
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(INSERT_SQL)) {
            
            pstmt.setString(1, customer.getCustomerId());
            pstmt.setString(2, customer.getFirstName());
            pstmt.setString(3, customer.getLastName());
            pstmt.setString(4, customer.getEmail());
            pstmt.setString(5, customer.getPhone());
            pstmt.setDate(6, customer.getAccountCreatedDate() != null ? 
                    Date.valueOf(customer.getAccountCreatedDate()) : null);
            pstmt.setString(7, customer.getRiskLevel() != null ? customer.getRiskLevel().name() : null);
            pstmt.setInt(8, customer.getTotalTransactions());
            pstmt.setBigDecimal(9, customer.getLifetimeValue());
            pstmt.setTimestamp(10, Timestamp.valueOf(LocalDateTime.now()));
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully saved customer: {}", customer.getCustomerId());
            } else {
                logger.warn("Failed to save customer: {}", customer.getCustomerId());
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while saving customer {}: {}", 
                    customer.getCustomerId(), e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Finds a customer by their ID.
     * 
     * @param customerId the customer ID
     * @return Optional containing the customer if found
     */
    public Optional<Customer> findById(String customerId) {
        logger.debug("Finding customer by ID: {}", customerId);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_BY_ID_SQL)) {
            
            pstmt.setString(1, customerId);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    Customer customer = mapResultSetToCustomer(rs);
                    logger.debug("Found customer: {}", customerId);
                    return Optional.of(customer);
                }
            }
            
        } catch (SQLException e) {
            logger.error("Database error while finding customer {}: {}", customerId, e.getMessage(), e);
        }
        
        logger.debug("Customer not found: {}", customerId);
        return Optional.empty();
    }
    
    /**
     * Finds all customers in the database.
     * 
     * @return list of all customers
     */
    public List<Customer> findAll() {
        logger.debug("Finding all customers");
        
        List<Customer> customers = new ArrayList<>();
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_ALL_SQL);
             ResultSet rs = pstmt.executeQuery()) {
            
            while (rs.next()) {
                customers.add(mapResultSetToCustomer(rs));
            }
            
            logger.info("Found {} customers", customers.size());
            
        } catch (SQLException e) {
            logger.error("Database error while finding all customers: {}", e.getMessage(), e);
        }
        
        return customers;
    }
    
    /**
     * Finds customers by risk level.
     * 
     * @param riskLevel the risk level to filter by
     * @return list of customers with the specified risk level
     */
    public List<Customer> findByRiskLevel(String riskLevel) {
        logger.debug("Finding customers with risk level: {}", riskLevel);
        
        List<Customer> customers = new ArrayList<>();
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_BY_RISK_LEVEL_SQL)) {
            
            pstmt.setString(1, riskLevel);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    customers.add(mapResultSetToCustomer(rs));
                }
            }
            
            logger.info("Found {} customers with risk level {}", customers.size(), riskLevel);
            
        } catch (SQLException e) {
            logger.error("Database error while finding customers by risk level: {}", e.getMessage(), e);
        }
        
        return customers;
    }
    
    /**
     * Updates an existing customer.
     * 
     * @param customer the customer with updated information
     * @return true if update was successful
     */
    public boolean update(Customer customer) {
        logger.debug("Updating customer: {}", customer.getCustomerId());
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(UPDATE_SQL)) {
            
            pstmt.setString(1, customer.getFirstName());
            pstmt.setString(2, customer.getLastName());
            pstmt.setString(3, customer.getEmail());
            pstmt.setString(4, customer.getPhone());
            pstmt.setString(5, customer.getRiskLevel() != null ? customer.getRiskLevel().name() : null);
            pstmt.setInt(6, customer.getTotalTransactions());
            pstmt.setBigDecimal(7, customer.getLifetimeValue());
            pstmt.setTimestamp(8, customer.getLastTransactionDate() != null ? 
                    Timestamp.valueOf(customer.getLastTransactionDate()) : null);
            pstmt.setTimestamp(9, Timestamp.valueOf(LocalDateTime.now()));
            pstmt.setString(10, customer.getCustomerId());
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully updated customer: {}", customer.getCustomerId());
            } else {
                logger.warn("No customer found to update: {}", customer.getCustomerId());
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while updating customer: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Updates customer transaction statistics.
     * Used after processing a new transaction.
     * 
     * @param customerId the customer ID
     * @param transactionAmount the amount of the new transaction
     * @return true if update was successful
     */
    public boolean updateTransactionStats(String customerId, java.math.BigDecimal transactionAmount) {
        logger.debug("Updating transaction stats for customer: {}", customerId);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(UPDATE_TRANSACTION_STATS_SQL)) {
            
            pstmt.setBigDecimal(1, transactionAmount);
            pstmt.setTimestamp(2, Timestamp.valueOf(LocalDateTime.now()));
            pstmt.setTimestamp(3, Timestamp.valueOf(LocalDateTime.now()));
            pstmt.setString(4, customerId);
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully updated transaction stats for customer: {}", customerId);
            } else {
                logger.warn("No customer found to update stats: {}", customerId);
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while updating transaction stats: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Deletes a customer by ID.
     * 
     * @param customerId the customer ID
     * @return true if deletion was successful
     */
    public boolean delete(String customerId) {
        logger.debug("Deleting customer: {}", customerId);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(DELETE_SQL)) {
            
            pstmt.setString(1, customerId);
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully deleted customer {}", customerId);
            } else {
                logger.warn("No customer found to delete: {}", customerId);
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while deleting customer: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Maps a ResultSet row to a Customer object.
     */
    private Customer mapResultSetToCustomer(ResultSet rs) throws SQLException {
        Customer customer = new Customer();
        customer.setCustomerId(rs.getString("customer_id"));
        customer.setFirstName(rs.getString("first_name"));
        customer.setLastName(rs.getString("last_name"));
        customer.setEmail(rs.getString("email"));
        customer.setPhone(rs.getString("phone"));
        
        Date accountCreatedDate = rs.getDate("account_created_date");
        if (accountCreatedDate != null) {
            customer.setAccountCreatedDate(accountCreatedDate.toLocalDate());
        }
        
        customer.setRiskLevel(rs.getString("risk_level") != null ? 
                Customer.RiskLevel.valueOf(rs.getString("risk_level")) : null);
        customer.setTotalTransactions(rs.getInt("total_transactions"));
        customer.setLifetimeValue(rs.getBigDecimal("lifetime_value"));
        
        Timestamp lastTransactionDate = rs.getTimestamp("last_transaction_date");
        if (lastTransactionDate != null) {
            customer.setLastTransactionDate(lastTransactionDate.toLocalDateTime());
        }
        
        return customer;
    }
}
