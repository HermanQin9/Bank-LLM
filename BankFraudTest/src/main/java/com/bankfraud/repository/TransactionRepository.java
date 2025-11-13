package com.bankfraud.repository;

import com.bankfraud.config.DatabaseConfig;
import com.bankfraud.model.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.sql.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Repository for Transaction entity database operations.
 * Handles CRUD operations and queries for transactions.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
public class TransactionRepository {
    
    private static final Logger logger = LoggerFactory.getLogger(TransactionRepository.class);
    
    private static final String INSERT_SQL = 
            "INSERT INTO transactions (transaction_id, customer_id, amount, transaction_date, " +
            "merchant_name, merchant_category, location, fraud_flag, source_system, status, created_at) " +
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
    
    private static final String SELECT_BY_ID_SQL = 
            "SELECT * FROM transactions WHERE transaction_id = ?";
    
    private static final String SELECT_BY_CUSTOMER_SQL = 
            "SELECT * FROM transactions WHERE customer_id = ? ORDER BY transaction_date DESC";
    
    private static final String SELECT_BY_DATE_RANGE_SQL = 
            "SELECT * FROM transactions WHERE transaction_date BETWEEN ? AND ? ORDER BY transaction_date";
    
    private static final String SELECT_FRAUDULENT_SQL = 
            "SELECT * FROM transactions WHERE fraud_flag = true ORDER BY transaction_date DESC";
    
    private static final String UPDATE_STATUS_SQL = 
            "UPDATE transactions SET status = ?, updated_at = ? WHERE transaction_id = ?";
    
    private static final String DELETE_SQL = 
            "DELETE FROM transactions WHERE transaction_id = ?";
    
    private static final String COUNT_RECENT_TRANSACTIONS_SQL = 
            "SELECT COUNT(*) FROM transactions WHERE customer_id = ? " +
            "AND transaction_date >= ?";
    
    /**
     * Counts recent transactions for a customer within specified hours.
     * Used for velocity checks in fraud detection.
     * 
     * @param customerId the customer ID
     * @param hours number of hours to look back
     * @return count of transactions in the time window
     */
    public long countRecentTransactions(String customerId, int hours) {
        logger.debug("Counting transactions for customer {} in last {} hours", customerId, hours);
        
        LocalDateTime cutoffTime = LocalDateTime.now().minusHours(hours);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(COUNT_RECENT_TRANSACTIONS_SQL)) {
            
            pstmt.setString(1, customerId);
            pstmt.setTimestamp(2, Timestamp.valueOf(cutoffTime));
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    long count = rs.getLong(1);
                    logger.debug("Found {} recent transactions for customer {}", count, customerId);
                    return count;
                }
            }
            
        } catch (SQLException e) {
            logger.error("Database error while counting recent transactions for customer {}: {}", 
                    customerId, e.getMessage(), e);
        }
        
        return 0;
    }
    
    /**
     * Inserts a new transaction into the database.
     * 
     * @param transaction the transaction to insert
     * @return true if insert was successful
     */
    public boolean save(Transaction transaction) {
        logger.debug("Attempting to save transaction: {}", transaction.getTransactionId());
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(INSERT_SQL)) {
            
            pstmt.setString(1, transaction.getTransactionId());
            pstmt.setString(2, transaction.getCustomerId());
            pstmt.setBigDecimal(3, transaction.getAmount());
            pstmt.setTimestamp(4, Timestamp.valueOf(transaction.getTransactionDate()));
            pstmt.setString(5, transaction.getMerchantName());
            pstmt.setString(6, transaction.getMerchantCategory());
            pstmt.setString(7, transaction.getLocation());
            pstmt.setBoolean(8, transaction.isFraudFlag());
            pstmt.setString(9, transaction.getSourceSystem());
            pstmt.setString(10, transaction.getStatus());
            pstmt.setTimestamp(11, Timestamp.valueOf(LocalDateTime.now()));
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully saved transaction: {}", transaction.getTransactionId());
            } else {
                logger.warn("Failed to save transaction: {}", transaction.getTransactionId());
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while saving transaction {}: {}", 
                    transaction.getTransactionId(), e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Batch inserts multiple transactions.
     * More efficient than individual inserts.
     * 
     * @param transactions list of transactions to insert
     * @return number of successfully inserted transactions
     */
    public int saveBatch(List<Transaction> transactions) {
        logger.info("Starting batch insert of {} transactions", transactions.size());
        
        int successCount = 0;
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection()) {
            conn.setAutoCommit(false);
            
            try (PreparedStatement pstmt = conn.prepareStatement(INSERT_SQL)) {
                
                for (Transaction transaction : transactions) {
                    pstmt.setString(1, transaction.getTransactionId());
                    pstmt.setString(2, transaction.getCustomerId());
                    pstmt.setBigDecimal(3, transaction.getAmount());
                    pstmt.setTimestamp(4, Timestamp.valueOf(transaction.getTransactionDate()));
                    pstmt.setString(5, transaction.getMerchantName());
                    pstmt.setString(6, transaction.getMerchantCategory());
                    pstmt.setString(7, transaction.getLocation());
                    pstmt.setBoolean(8, transaction.isFraudFlag());
                    pstmt.setString(9, transaction.getSourceSystem());
                    pstmt.setString(10, transaction.getStatus());
                    pstmt.setTimestamp(11, Timestamp.valueOf(LocalDateTime.now()));
                    
                    pstmt.addBatch();
                    
                    // Execute batch every 1000 records
                    if ((successCount + 1) % 1000 == 0) {
                        int[] results = pstmt.executeBatch();
                        successCount += countSuccessful(results);
                        logger.debug("Batch insert progress: {} transactions", successCount);
                    }
                }
                
                // Execute remaining batch
                int[] results = pstmt.executeBatch();
                successCount += countSuccessful(results);
                
                conn.commit();
                logger.info("Successfully batch inserted {} transactions", successCount);
                
            } catch (SQLException e) {
                conn.rollback();
                logger.error("Batch insert failed, rolled back. Error: {}", e.getMessage(), e);
                throw e;
            }
            
        } catch (SQLException e) {
            logger.error("Database error during batch insert: {}", e.getMessage(), e);
        }
        
        return successCount;
    }
    
    /**
     * Counts successful batch operations.
     */
    private int countSuccessful(int[] results) {
        int count = 0;
        for (int result : results) {
            if (result > 0) count++;
        }
        return count;
    }
    
    /**
     * Finds a transaction by its ID.
     * 
     * @param transactionId the transaction ID
     * @return Optional containing the transaction if found
     */
    public Optional<Transaction> findById(String transactionId) {
        logger.debug("Finding transaction by ID: {}", transactionId);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_BY_ID_SQL)) {
            
            pstmt.setString(1, transactionId);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                if (rs.next()) {
                    Transaction transaction = mapResultSetToTransaction(rs);
                    logger.debug("Found transaction: {}", transactionId);
                    return Optional.of(transaction);
                }
            }
            
        } catch (SQLException e) {
            logger.error("Database error while finding transaction {}: {}", transactionId, e.getMessage(), e);
        }
        
        logger.debug("Transaction not found: {}", transactionId);
        return Optional.empty();
    }
    
    /**
     * Finds all transactions for a specific customer.
     * 
     * @param customerId the customer ID
     * @return list of transactions
     */
    public List<Transaction> findByCustomerId(String customerId) {
        logger.debug("Finding transactions for customer: {}", customerId);
        
        List<Transaction> transactions = new ArrayList<>();
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_BY_CUSTOMER_SQL)) {
            
            pstmt.setString(1, customerId);
            
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    transactions.add(mapResultSetToTransaction(rs));
                }
            }
            
            logger.info("Found {} transactions for customer {}", transactions.size(), customerId);
            
        } catch (SQLException e) {
            logger.error("Database error while finding transactions for customer {}: {}", 
                    customerId, e.getMessage(), e);
        }
        
        return transactions;
    }
    
    /**
     * Finds transactions within a date range.
     * 
     * @param startDate start of date range
     * @param endDate end of date range
     * @return list of transactions
     */
    public List<Transaction> findByDateRange(LocalDateTime startDate, LocalDateTime endDate) {
        logger.debug("Finding transactions between {} and {}", startDate, endDate);
        
        List<Transaction> transactions = new ArrayList<>();
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_BY_DATE_RANGE_SQL)) {
            
            pstmt.setTimestamp(1, Timestamp.valueOf(startDate));
            pstmt.setTimestamp(2, Timestamp.valueOf(endDate));
            
            try (ResultSet rs = pstmt.executeQuery()) {
                while (rs.next()) {
                    transactions.add(mapResultSetToTransaction(rs));
                }
            }
            
            logger.info("Found {} transactions in date range", transactions.size());
            
        } catch (SQLException e) {
            logger.error("Database error while finding transactions by date range: {}", e.getMessage(), e);
        }
        
        return transactions;
    }
    
    /**
     * Finds all fraudulent transactions.
     * 
     * @return list of fraudulent transactions
     */
    public List<Transaction> findFraudulent() {
        logger.debug("Finding fraudulent transactions");
        
        List<Transaction> transactions = new ArrayList<>();
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(SELECT_FRAUDULENT_SQL);
             ResultSet rs = pstmt.executeQuery()) {
            
            while (rs.next()) {
                transactions.add(mapResultSetToTransaction(rs));
            }
            
            logger.info("Found {} fraudulent transactions", transactions.size());
            
        } catch (SQLException e) {
            logger.error("Database error while finding fraudulent transactions: {}", e.getMessage(), e);
        }
        
        return transactions;
    }
    
    /**
     * Updates the status of a transaction.
     * 
     * @param transactionId the transaction ID
     * @param newStatus the new status
     * @return true if update was successful
     */
    public boolean updateStatus(String transactionId, String newStatus) {
        logger.debug("Updating status for transaction {} to {}", transactionId, newStatus);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(UPDATE_STATUS_SQL)) {
            
            pstmt.setString(1, newStatus);
            pstmt.setTimestamp(2, Timestamp.valueOf(LocalDateTime.now()));
            pstmt.setString(3, transactionId);
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully updated status for transaction {}", transactionId);
            } else {
                logger.warn("No transaction found to update: {}", transactionId);
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while updating transaction status: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Deletes a transaction by ID.
     * 
     * @param transactionId the transaction ID
     * @return true if deletion was successful
     */
    public boolean delete(String transactionId) {
        logger.debug("Deleting transaction: {}", transactionId);
        
        try (Connection conn = DatabaseConfig.getDataSource().getConnection();
             PreparedStatement pstmt = conn.prepareStatement(DELETE_SQL)) {
            
            pstmt.setString(1, transactionId);
            
            int rowsAffected = pstmt.executeUpdate();
            boolean success = rowsAffected > 0;
            
            if (success) {
                logger.info("Successfully deleted transaction {}", transactionId);
            } else {
                logger.warn("No transaction found to delete: {}", transactionId);
            }
            
            return success;
            
        } catch (SQLException e) {
            logger.error("Database error while deleting transaction: {}", e.getMessage(), e);
            return false;
        }
    }
    
    /**
     * Maps a ResultSet row to a Transaction object.
     */
    private Transaction mapResultSetToTransaction(ResultSet rs) throws SQLException {
        Transaction transaction = new Transaction();
        transaction.setTransactionId(rs.getString("transaction_id"));
        transaction.setCustomerId(rs.getString("customer_id"));
        transaction.setAmount(rs.getBigDecimal("amount"));
        transaction.setTransactionDate(rs.getTimestamp("transaction_date").toLocalDateTime());
        transaction.setMerchantName(rs.getString("merchant_name"));
        transaction.setMerchantCategory(rs.getString("merchant_category"));
        transaction.setLocation(rs.getString("location"));
        transaction.setFraudFlag(rs.getBoolean("fraud_flag"));
        transaction.setSourceSystem(rs.getString("source_system"));
        transaction.setStatus(rs.getString("status"));
        return transaction;
    }
}
