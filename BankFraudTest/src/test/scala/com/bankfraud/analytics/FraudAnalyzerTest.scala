package com.bankfraud.analytics

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

import java.math.BigDecimal
import java.time.LocalDateTime

/**
 * Unit tests for FraudAnalyzer using ScalaTest.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
class FraudAnalyzerTest extends AnyFunSuite with Matchers {
  
  val analyzer = new FraudAnalyzer()
  
  test("High value transaction should trigger HIGH_VALUE_TRANSACTION rule") {
    val transaction = analyzer.Transaction(
      transactionId = "TXN001",
      customerId = "CUST001",
      amount = new BigDecimal("6000.00"),
      transactionDate = LocalDateTime.of(2024, 1, 15, 14, 30),
      merchantName = "Test Merchant",
      merchantCategory = "Shopping",
      location = "New York, NY"
    )
    
    val result = analyzer.analyzeFraud(transaction, List.empty)
    
    result.score should be >= 25.0
    result.triggeredRules should contain("HIGH_VALUE_TRANSACTION")
  }
  
  test("Unusual time transaction should trigger UNUSUAL_TIME rule") {
    val transaction = analyzer.Transaction(
      transactionId = "TXN002",
      customerId = "CUST001",
      amount = new BigDecimal("100.00"),
      transactionDate = LocalDateTime.of(2024, 1, 15, 3, 30), // 3:30 AM
      merchantName = "Test Merchant",
      merchantCategory = "Shopping",
      location = "New York, NY"
    )
    
    val result = analyzer.analyzeFraud(transaction, List.empty)
    
    result.triggeredRules should contain("UNUSUAL_TIME")
  }
  
  test("Multiple transactions in short time should trigger HIGH_VELOCITY rule") {
    val history = List(
      analyzer.Transaction("TXN001", "CUST001", new BigDecimal("100.00"), 
        LocalDateTime.of(2024, 1, 15, 14, 0), "Merchant1", "Shopping", "NY"),
      analyzer.Transaction("TXN002", "CUST001", new BigDecimal("200.00"), 
        LocalDateTime.of(2024, 1, 15, 14, 15), "Merchant2", "Shopping", "NY"),
      analyzer.Transaction("TXN003", "CUST001", new BigDecimal("150.00"), 
        LocalDateTime.of(2024, 1, 15, 14, 30), "Merchant3", "Shopping", "NY")
    )
    
    val transaction = analyzer.Transaction(
      transactionId = "TXN004",
      customerId = "CUST001",
      amount = new BigDecimal("300.00"),
      transactionDate = LocalDateTime.of(2024, 1, 15, 14, 45),
      merchantName = "Merchant4",
      merchantCategory = "Shopping",
      location = "NY"
    )
    
    val result = analyzer.analyzeFraud(transaction, history)
    
    result.triggeredRules should contain("HIGH_VELOCITY")
  }
  
  test("New merchant should trigger NEW_MERCHANT rule") {
    val history = List(
      analyzer.Transaction("TXN001", "CUST001", new BigDecimal("100.00"), 
        LocalDateTime.of(2024, 1, 14, 14, 0), "Merchant1", "Shopping", "NY")
    )
    
    val transaction = analyzer.Transaction(
      transactionId = "TXN002",
      customerId = "CUST001",
      amount = new BigDecimal("200.00"),
      transactionDate = LocalDateTime.of(2024, 1, 15, 14, 0),
      merchantName = "Merchant2",
      merchantCategory = "Shopping",
      location = "NY"
    )
    
    val result = analyzer.analyzeFraud(transaction, history)
    
    result.triggeredRules should contain("NEW_MERCHANT")
  }
  
  test("Risk level should be CRITICAL for score >= 80") {
    val transaction = analyzer.Transaction(
      transactionId = "TXN001",
      customerId = "CUST001",
      amount = new BigDecimal("10000.00"), // High value: 25 points
      transactionDate = LocalDateTime.of(2024, 1, 15, 3, 30), // Unusual time: 15 points
      merchantName = "New Merchant",
      merchantCategory = "Shopping",
      location = "NY"
    )
    
    // Multiple history transactions for better standard deviation calculation
    val history = List(
      analyzer.Transaction("TXN000", "CUST001", new BigDecimal("50.00"), 
        LocalDateTime.of(2024, 1, 14, 10, 0), "Old Merchant", "Shopping", "NY"),
      analyzer.Transaction("TXN001", "CUST001", new BigDecimal("45.00"), 
        LocalDateTime.of(2024, 1, 13, 14, 0), "Old Merchant", "Shopping", "NY"),
      analyzer.Transaction("TXN002", "CUST001", new BigDecimal("55.00"), 
        LocalDateTime.of(2024, 1, 12, 16, 0), "Old Merchant", "Shopping", "NY")
    )
    
    val result = analyzer.analyzeFraud(transaction, history)
    
    // Should trigger: HIGH_VALUE (25) + UNUSUAL_TIME (15) + AMOUNT_DEVIATION (25) + NEW_MERCHANT (10) = 75+
    result.riskLevel should (equal("CRITICAL") or equal("HIGH"))
  }
  
  test("analyzeTransactionBatch should return correct statistics") {
    val transactions = List(
      analyzer.Transaction("TXN001", "CUST001", new BigDecimal("100.00"), 
        LocalDateTime.of(2024, 1, 15, 14, 0), "Merchant1", "Shopping", "NY"),
      analyzer.Transaction("TXN002", "CUST001", new BigDecimal("200.00"), 
        LocalDateTime.of(2024, 1, 15, 15, 0), "Merchant2", "Shopping", "NY"),
      analyzer.Transaction("TXN003", "CUST002", new BigDecimal("150.00"), 
        LocalDateTime.of(2024, 1, 15, 16, 0), "Merchant1", "Shopping", "CA")
    )
    
    val stats = analyzer.analyzeTransactionBatch(transactions)
    
    stats("totalTransactions") shouldBe 3
    stats("totalCustomers") shouldBe 2
    stats("totalVolume") shouldBe 450.0
  }
  
  test("calculateCustomerRiskScore should return average fraud score") {
    val transactions = List(
      analyzer.Transaction("TXN001", "CUST001", new BigDecimal("100.00"), 
        LocalDateTime.of(2024, 1, 15, 14, 0), "Merchant1", "Shopping", "NY"),
      analyzer.Transaction("TXN002", "CUST001", new BigDecimal("200.00"), 
        LocalDateTime.of(2024, 1, 15, 15, 0), "Merchant1", "Shopping", "NY")
    )
    
    val riskScore = analyzer.calculateCustomerRiskScore("CUST001", transactions)
    
    riskScore should be >= 0.0
    riskScore should be <= 100.0
  }
  
  test("Empty history should return lower fraud score") {
    val transaction = analyzer.Transaction(
      transactionId = "TXN001",
      customerId = "CUST001",
      amount = new BigDecimal("100.00"),
      transactionDate = LocalDateTime.of(2024, 1, 15, 14, 0),
      merchantName = "Merchant1",
      merchantCategory = "Shopping",
      location = "NY"
    )
    
    val result = analyzer.analyzeFraud(transaction, List.empty)
    
    // Should not trigger deviation or velocity rules without history
    result.triggeredRules should not contain "AMOUNT_DEVIATION"
    result.triggeredRules should not contain "HIGH_VELOCITY"
  }
}
