package com.bankfraud.spark

import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.BeforeAndAfterAll
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import java.sql.Timestamp
import java.time.LocalDateTime

/**
 * Test suite for Spark Transaction Processor
 * 
 * Tests:
 * - Spark session creation
 * - Data loading and transformation
 * - Fraud detection algorithms
 * - S3 integration (mocked)
 * 
 * @author Banking Platform Team
 */
class SparkTransactionProcessorTest extends AnyFunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll(): Unit = {
    // Create local Spark session for testing
    spark = SparkSession.builder()
      .appName("BankFraud-Test")
      .master("local[2]")
      .config("spark.sql.shuffle.partitions", "2")
      .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
  }

  override def afterAll(): Unit = {
    if (spark != null) {
      spark.stop()
    }
  }

  def createTestTransactions(): DataFrame = {
    val now = Timestamp.valueOf(LocalDateTime.now())
    val oneHourAgo = Timestamp.valueOf(LocalDateTime.now().minusHours(1))
    
    spark.createDataFrame(Seq(
      ("TXN001", "CUST001", oneHourAgo, 1000.0, "USD", "Merchant A", "RETAIL", "PURCHASE", "US", "New York", true),
      ("TXN002", "CUST001", oneHourAgo, 2000.0, "USD", "Merchant B", "RETAIL", "PURCHASE", "US", "New York", true),
      ("TXN003", "CUST001", now, 8000.0, "USD", "Merchant C", "WIRE_TRANSFER", "WIRE", "CN", "Shanghai", true),
      ("TXN004", "CUST002", now, 500.0, "USD", "Merchant A", "RETAIL", "PURCHASE", "US", "Boston", false),
      ("TXN005", "CUST003", Timestamp.valueOf(LocalDateTime.now().withHour(3)), 15000.0, "USD", "Unknown", "ATM", "CASH", "RU", "Moscow", false)
    )).toDF(
      "transaction_id", "customer_id", "transaction_date", "amount", "currency",
      "merchant_name", "merchant_category", "transaction_type", "location_country",
      "location_city", "is_online"
    )
  }

  test("Spark session should be created successfully") {
    assert(spark != null)
    assert(spark.sparkContext.master == "local[2]")
  }

  test("Test transactions DataFrame should be created") {
    val transactions = createTestTransactions()
    
    assert(transactions.count() == 5)
    assert(transactions.columns.length == 11)
    assert(transactions.columns.contains("transaction_id"))
    assert(transactions.columns.contains("customer_id"))
  }

  test("High-value transaction detection should work") {
    val transactions = createTestTransactions()
    val highValueTxns = SparkTransactionProcessor.detectHighValueTransactions(transactions)
    
    // Should detect transactions > $5000
    assert(highValueTxns.count() == 2) // TXN003: $8000, TXN005: $15000
    
    val amounts = highValueTxns.select("amount").collect().map(_.getDouble(0))
    assert(amounts.forall(_ > 5000.0))
  }

  test("Velocity anomaly detection should work") {
    val transactions = createTestTransactions()
    val velocityAlerts = SparkTransactionProcessor.detectVelocityAnomalies(transactions, 60, 3)
    
    // CUST001 has 3 transactions within 1 hour window
    assert(velocityAlerts.count() >= 1)
    
    val customerIds = velocityAlerts.select("customer_id").distinct().collect().map(_.getString(0))
    assert(customerIds.contains("CUST001"))
  }

  test("Statistical anomaly detection should identify outliers") {
    val transactions = createTestTransactions()
    val statisticalAlerts = SparkTransactionProcessor.detectStatisticalAnomalies(transactions)
    
    // High z-score transactions should be detected
    assert(statisticalAlerts.count() > 0)
    
    val zScores = statisticalAlerts.select("z_score").collect().map(_.getDouble(0))
    assert(zScores.forall(_ >= 2.0))
  }

  test("Unusual time pattern detection should work") {
    val transactions = createTestTransactions()
    val unusualTimeAlerts = SparkTransactionProcessor.detectUnusualTimePatterns(transactions)
    
    // TXN005 occurs at 3 AM
    assert(unusualTimeAlerts.count() >= 1)
    
    val hours = unusualTimeAlerts.withColumn("hour", hour(col("transaction_date")))
      .select("hour").collect().map(_.getInt(0))
    assert(hours.forall(h => h >= 2 && h <= 5))
  }

  test("Geographic anomaly detection should identify impossible travel") {
    val transactions = createTestTransactions()
    val geoAlerts = SparkTransactionProcessor.detectGeographicAnomalies(transactions)
    
    // CUST001: US -> China in short time = impossible
    assert(geoAlerts.count() >= 1)
    
    val timeDiffs = geoAlerts.select("time_diff_hours").collect().map(_.getDouble(0))
    assert(timeDiffs.forall(_ < 6.0))
  }

  test("Fraud detection pipeline should combine all detection methods") {
    val transactions = createTestTransactions()
    val fraudAlerts = SparkTransactionProcessor.runFraudDetectionPipeline(transactions)
    
    assert(fraudAlerts.count() > 0)
    assert(fraudAlerts.columns.contains("total_risk_score"))
    assert(fraudAlerts.columns.contains("risk_level"))
    assert(fraudAlerts.columns.contains("risk_factors"))
    
    // Check risk levels
    val riskLevels = fraudAlerts.select("risk_level").distinct().collect().map(_.getString(0))
    assert(riskLevels.forall(level => Seq("LOW", "MEDIUM", "HIGH", "CRITICAL").contains(level)))
  }

  test("Customer risk profiling should generate aggregate statistics") {
    val transactions = createTestTransactions()
    val customerProfiles = SparkTransactionProcessor.generateCustomerRiskProfiles(transactions)
    
    assert(customerProfiles.count() == 3) // 3 distinct customers
    assert(customerProfiles.columns.contains("total_transactions"))
    assert(customerProfiles.columns.contains("total_volume"))
    assert(customerProfiles.columns.contains("customer_risk_score"))
    
    // CUST001 should have 3 transactions
    val cust001 = customerProfiles.filter(col("customer_id") === "CUST001").collect()
    assert(cust001.length == 1)
    assert(cust001(0).getAs[Long]("total_transactions") == 3)
  }

  test("Risk score calculation should be within valid range") {
    val transactions = createTestTransactions()
    val fraudAlerts = SparkTransactionProcessor.runFraudDetectionPipeline(transactions)
    
    val riskScores = fraudAlerts.select("total_risk_score").collect().map(_.getDouble(0))
    assert(riskScores.forall(score => score >= 0.0 && score <= 100.0))
  }

  test("Multiple fraud indicators should accumulate risk score") {
    val transactions = createTestTransactions()
    val fraudAlerts = SparkTransactionProcessor.runFraudDetectionPipeline(transactions)
    
    // Transaction with multiple risk factors should have higher score
    val highRiskTxn = fraudAlerts.orderBy(col("total_risk_score").desc).first()
    val riskFactors = highRiskTxn.getAs[Seq[String]]("risk_factors")
    
    // Should have at least 2 risk factors
    assert(riskFactors.length >= 2)
  }

  test("DataFrame transformations should preserve data integrity") {
    val transactions = createTestTransactions()
    val originalCount = transactions.count()
    
    // Apply transformation
    val withRiskScore = transactions.withColumn("risk_score", lit(0.0))
    
    assert(withRiskScore.count() == originalCount)
    assert(withRiskScore.columns.length == transactions.columns.length + 1)
  }
}
