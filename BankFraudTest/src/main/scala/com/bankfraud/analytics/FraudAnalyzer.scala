package com.bankfraud.analytics

import java.math.BigDecimal
import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import scala.collection.mutable

/**
 * Fraud detection analyzer using Scala for advanced analytics.
 * Demonstrates functional programming approach to fraud detection.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
class FraudAnalyzer {
  
  /**
   * Transaction case class for Scala processing.
   */
  case class Transaction(
    transactionId: String,
    customerId: String,
    amount: BigDecimal,
    transactionDate: LocalDateTime,
    merchantName: String,
    merchantCategory: String,
    location: String
  )
  
  /**
   * Fraud detection result.
   */
  case class FraudScore(
    transactionId: String,
    score: Double,
    riskLevel: String,
    triggeredRules: List[String]
  )
  
  /**
   * Analyzes a transaction for fraud indicators.
   * 
   * @param transaction the transaction to analyze
   * @param customerHistory historical transactions for the customer
   * @return fraud score with risk level
   */
  def analyzeFraud(
    transaction: Transaction,
    customerHistory: List[Transaction]
  ): FraudScore = {
    
    val triggeredRules = mutable.ListBuffer[String]()
    var score = 0.0
    
    // Rule 1: High-value transaction
    if (isHighValue(transaction.amount)) {
      score += 25.0
      triggeredRules += "HIGH_VALUE_TRANSACTION"
    }
    
    // Rule 2: Unusual time
    if (isUnusualTime(transaction.transactionDate)) {
      score += 15.0
      triggeredRules += "UNUSUAL_TIME"
    }
    
    // Rule 3: Velocity check (multiple transactions in short time)
    val velocityScore = checkVelocity(transaction, customerHistory)
    score += velocityScore
    if (velocityScore > 0) {
      triggeredRules += "HIGH_VELOCITY"
    }
    
    // Rule 4: Amount deviation from customer average
    val deviationScore = checkAmountDeviation(transaction, customerHistory)
    score += deviationScore
    if (deviationScore > 0) {
      triggeredRules += "AMOUNT_DEVIATION"
    }
    
    // Rule 5: New merchant for customer
    if (isNewMerchant(transaction, customerHistory)) {
      score += 10.0
      triggeredRules += "NEW_MERCHANT"
    }
    
    // Determine risk level
    val riskLevel = score match {
      case s if s >= 80 => "CRITICAL"
      case s if s >= 60 => "HIGH"
      case s if s >= 40 => "MEDIUM"
      case s if s >= 20 => "LOW"
      case _ => "MINIMAL"
    }
    
    FraudScore(transaction.transactionId, score, riskLevel, triggeredRules.toList)
  }
  
  /**
   * Checks if transaction amount is high value (>$5000).
   */
  private def isHighValue(amount: BigDecimal): Boolean = {
    amount.compareTo(new BigDecimal("5000.00")) > 0
  }
  
  /**
   * Checks if transaction occurred during unusual hours (2 AM - 5 AM).
   */
  private def isUnusualTime(dateTime: LocalDateTime): Boolean = {
    val hour = dateTime.getHour
    hour >= 2 && hour < 5
  }
  
  /**
   * Checks transaction velocity (multiple transactions in 1 hour).
   * Returns score based on velocity.
   */
  private def checkVelocity(
    transaction: Transaction,
    history: List[Transaction]
  ): Double = {
    
    val recentTransactions = history.filter { t =>
      val hoursDiff = ChronoUnit.HOURS.between(t.transactionDate, transaction.transactionDate)
      hoursDiff >= 0 && hoursDiff <= 1
    }
    
    recentTransactions.length match {
      case n if n >= 5 => 30.0
      case n if n >= 3 => 20.0
      case n if n >= 2 => 10.0
      case _ => 0.0
    }
  }
  
  /**
   * Checks if amount deviates significantly from customer's average.
   * Returns score based on deviation.
   */
  private def checkAmountDeviation(
    transaction: Transaction,
    history: List[Transaction]
  ): Double = {
    
    if (history.isEmpty) return 0.0
    
    val amounts = history.map(_.amount.doubleValue())
    val average = amounts.sum / amounts.length
    val stdDev = math.sqrt(amounts.map(a => math.pow(a - average, 2)).sum / amounts.length)
    
    val deviation = math.abs(transaction.amount.doubleValue() - average)
    
    if (stdDev > 0) {
      val zScore = deviation / stdDev
      zScore match {
        case z if z >= 3 => 25.0
        case z if z >= 2 => 15.0
        case z if z >= 1.5 => 10.0
        case _ => 0.0
      }
    } else {
      0.0
    }
  }
  
  /**
   * Checks if merchant is new for the customer.
   */
  private def isNewMerchant(
    transaction: Transaction,
    history: List[Transaction]
  ): Boolean = {
    !history.exists(_.merchantName == transaction.merchantName)
  }
  
  /**
   * Analyzes multiple transactions and returns statistics.
   * Demonstrates functional programming with Scala collections.
   * 
   * @param transactions list of transactions to analyze
   * @return fraud statistics
   */
  def analyzeTransactionBatch(
    transactions: List[Transaction]
  ): Map[String, Any] = {
    
    val totalTransactions = transactions.length
    
    // Group by customer
    val byCustomer = transactions.groupBy(_.customerId)
    val totalCustomers = byCustomer.size
    
    // Calculate total volume
    val totalVolume = transactions.map(_.amount.doubleValue()).sum
    
    // High-value transactions
    val highValueCount = transactions.count(t => isHighValue(t.amount))
    
    // Transactions by hour
    val byHour = transactions.groupBy(_.transactionDate.getHour)
    val nightTransactions = byHour.filter { case (hour, _) => hour >= 0 && hour < 6 }.values.flatten.size
    
    // Top merchants
    val topMerchants = transactions
      .groupBy(_.merchantName)
      .mapValues(_.length)
      .toSeq
      .sortBy(-_._2)
      .take(5)
    
    Map(
      "totalTransactions" -> totalTransactions,
      "totalCustomers" -> totalCustomers,
      "totalVolume" -> totalVolume,
      "highValueTransactions" -> highValueCount,
      "nightTimeTransactions" -> nightTransactions,
      "topMerchants" -> topMerchants.toMap
    )
  }
  
  /**
   * Calculates risk score for a customer based on their transaction history.
   * 
   * @param customerId the customer ID
   * @param transactions all customer transactions
   * @return overall risk score (0-100)
   */
  def calculateCustomerRiskScore(
    customerId: String,
    transactions: List[Transaction]
  ): Double = {
    
    if (transactions.isEmpty) return 0.0
    
    val fraudScores = transactions.map { transaction =>
      val history = transactions.filter(t => 
        t.transactionDate.isBefore(transaction.transactionDate)
      )
      analyzeFraud(transaction, history).score
    }
    
    // Average fraud score
    fraudScores.sum / fraudScores.length
  }
}

/**
 * Companion object with utility functions.
 */
object FraudAnalyzer {
  
  /**
   * Factory method to create FraudAnalyzer instance.
   */
  def apply(): FraudAnalyzer = new FraudAnalyzer()
}
