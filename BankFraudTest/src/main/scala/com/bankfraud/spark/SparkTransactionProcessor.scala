package com.bankfraud.spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import java.util.Properties

/**
 * Apache Spark Batch Processor for Large-Scale Transaction Analysis
 * 
 * This module demonstrates production-ready Spark implementation for:
 * - Processing millions of transactions from S3/HDFS
 * - Advanced aggregations and fraud pattern detection
 * - Integration with existing fraud detection logic
 * - Optimized for AWS EMR deployment
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
object SparkTransactionProcessor {

  /**
   * Transaction schema definition
   */
  val transactionSchema: StructType = StructType(Array(
    StructField("transaction_id", StringType, nullable = false),
    StructField("customer_id", StringType, nullable = false),
    StructField("transaction_date", TimestampType, nullable = false),
    StructField("amount", DecimalType(12, 2), nullable = false),
    StructField("currency", StringType, nullable = true),
    StructField("merchant_name", StringType, nullable = true),
    StructField("merchant_category", StringType, nullable = true),
    StructField("transaction_type", StringType, nullable = true),
    StructField("location_country", StringType, nullable = true),
    StructField("location_city", StringType, nullable = true),
    StructField("is_online", BooleanType, nullable = true)
  ))

  /**
   * Initialize SparkSession with optimized configurations for EMR
   */
  def createSparkSession(appName: String = "BankFraudDetection"): SparkSession = {
    SparkSession.builder()
      .appName(appName)
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .config("spark.sql.shuffle.partitions", "200")
      .config("spark.sql.parquet.compression.codec", "snappy")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "512m")
      // EMR-specific optimizations
      .config("spark.speculation", "true")
      .config("spark.sql.broadcastTimeout", "600")
      .getOrCreate()
  }

  /**
   * Read transactions from multiple sources (S3, HDFS, JDBC)
   * 
   * @param spark SparkSession
   * @param sourcePath Path to data source (s3://bucket/path or jdbc:...)
   * @param format Data format (parquet, csv, json, jdbc)
   * @return DataFrame of transactions
   */
  def readTransactions(
    spark: SparkSession,
    sourcePath: String,
    format: String = "parquet"
  ): DataFrame = {
    
    format.toLowerCase match {
      case "parquet" =>
        spark.read
          .schema(transactionSchema)
          .parquet(sourcePath)
      
      case "csv" =>
        spark.read
          .option("header", "true")
          .option("inferSchema", "true")
          .option("dateFormat", "yyyy-MM-dd HH:mm:ss")
          .csv(sourcePath)
      
      case "json" =>
        spark.read
          .schema(transactionSchema)
          .json(sourcePath)
      
      case "jdbc" =>
        // Read from PostgreSQL
        val jdbcUrl = sys.env.getOrElse("DB_URL", "jdbc:postgresql://localhost:5432/frauddb")
        val connectionProperties = new Properties()
        connectionProperties.put("user", sys.env.getOrElse("DB_USER", "postgres"))
        connectionProperties.put("password", sys.env.getOrElse("DB_PASSWORD", "postgres"))
        connectionProperties.put("driver", "org.postgresql.Driver")
        
        spark.read
          .jdbc(jdbcUrl, "transactions", connectionProperties)
      
      case _ =>
        throw new IllegalArgumentException(s"Unsupported format: $format")
    }
  }

  /**
   * High-value transaction detection (>$5000)
   */
  def detectHighValueTransactions(transactions: DataFrame): DataFrame = {
    transactions
      .filter(col("amount") > 5000.0)
      .withColumn("risk_factor", lit("HIGH_VALUE"))
      .withColumn("risk_score", lit(25.0))
  }

  /**
   * Velocity analysis - detect multiple transactions in short time window
   */
  def detectVelocityAnomalies(
    transactions: DataFrame,
    timeWindowMinutes: Int = 60,
    threshold: Int = 3
  ): DataFrame = {
    
    import org.apache.spark.sql.expressions.Window
    
    val windowSpec = Window
      .partitionBy("customer_id")
      .orderBy(col("transaction_date").cast("long"))
      .rangeBetween(-timeWindowMinutes * 60, 0)
    
    transactions
      .withColumn("transaction_count_1h", count("*").over(windowSpec))
      .filter(col("transaction_count_1h") >= threshold)
      .withColumn("risk_factor", lit("HIGH_VELOCITY"))
      .withColumn("risk_score", 
        when(col("transaction_count_1h") >= 5, 30.0)
        .when(col("transaction_count_1h") >= 3, 20.0)
        .otherwise(10.0)
      )
  }

  /**
   * Statistical anomaly detection using z-score
   */
  def detectStatisticalAnomalies(transactions: DataFrame): DataFrame = {
    
    // Calculate customer statistics
    val customerStats = transactions
      .groupBy("customer_id")
      .agg(
        avg("amount").as("avg_amount"),
        stddev("amount").as("stddev_amount"),
        count("*").as("transaction_count")
      )
    
    // Join back and calculate z-scores
    transactions
      .join(customerStats, "customer_id")
      .withColumn("z_score", 
        when(col("stddev_amount") > 0,
          abs(col("amount") - col("avg_amount")) / col("stddev_amount")
        ).otherwise(0.0)
      )
      .filter(col("z_score") >= 2.0)
      .withColumn("risk_factor", lit("STATISTICAL_ANOMALY"))
      .withColumn("risk_score",
        when(col("z_score") >= 3, 25.0)
        .when(col("z_score") >= 2, 15.0)
        .otherwise(10.0)
      )
  }

  /**
   * Unusual time pattern detection (2 AM - 5 AM)
   */
  def detectUnusualTimePatterns(transactions: DataFrame): DataFrame = {
    transactions
      .withColumn("hour_of_day", hour(col("transaction_date")))
      .filter(col("hour_of_day").between(2, 5))
      .withColumn("risk_factor", lit("UNUSUAL_TIME"))
      .withColumn("risk_score", lit(15.0))
  }

  /**
   * Geographic anomaly detection
   */
  def detectGeographicAnomalies(transactions: DataFrame): DataFrame = {
    
    import org.apache.spark.sql.expressions.Window
    
    val windowSpec = Window
      .partitionBy("customer_id")
      .orderBy(col("transaction_date"))
    
    transactions
      .withColumn("prev_country", lag("location_country", 1).over(windowSpec))
      .withColumn("prev_city", lag("location_city", 1).over(windowSpec))
      .withColumn("prev_time", lag("transaction_date", 1).over(windowSpec))
      .withColumn("time_diff_hours", 
        (unix_timestamp(col("transaction_date")) - unix_timestamp(col("prev_time"))) / 3600
      )
      .filter(
        col("location_country") =!= col("prev_country") &&
        col("time_diff_hours") < 6  // Impossible to travel in 6 hours
      )
      .withColumn("risk_factor", lit("GEOGRAPHIC_ANOMALY"))
      .withColumn("risk_score", lit(20.0))
  }

  /**
   * Comprehensive fraud detection pipeline
   * Combines all detection methods
   */
  def runFraudDetectionPipeline(transactions: DataFrame): DataFrame = {
    
    // Apply all detection methods
    val highValue = detectHighValueTransactions(transactions)
    val velocity = detectVelocityAnomalies(transactions)
    val statistical = detectStatisticalAnomalies(transactions)
    val unusualTime = detectUnusualTimePatterns(transactions)
    val geographic = detectGeographicAnomalies(transactions)
    
    // Union all alerts
    val allAlerts = highValue
      .union(velocity)
      .union(statistical)
      .union(unusualTime)
      .union(geographic)
    
    // Aggregate risk scores per transaction
    allAlerts
      .groupBy(
        "transaction_id", "customer_id", "transaction_date", 
        "amount", "merchant_name"
      )
      .agg(
        sum("risk_score").as("total_risk_score"),
        collect_set("risk_factor").as("risk_factors")
      )
      .withColumn("risk_level",
        when(col("total_risk_score") >= 80, "CRITICAL")
        .when(col("total_risk_score") >= 60, "HIGH")
        .when(col("total_risk_score") >= 40, "MEDIUM")
        .otherwise("LOW")
      )
      .orderBy(col("total_risk_score").desc)
  }

  /**
   * Customer risk profiling - aggregate view
   */
  def generateCustomerRiskProfiles(transactions: DataFrame): DataFrame = {
    
    transactions
      .groupBy("customer_id")
      .agg(
        count("*").as("total_transactions"),
        sum("amount").as("total_volume"),
        avg("amount").as("avg_transaction"),
        max("amount").as("max_transaction"),
        min("amount").as("min_transaction"),
        stddev("amount").as("stddev_transaction"),
        countDistinct("merchant_name").as("unique_merchants"),
        countDistinct("location_country").as("countries_visited"),
        min("transaction_date").as("first_transaction"),
        max("transaction_date").as("last_transaction")
      )
      .withColumn("customer_risk_score",
        // Simple scoring logic
        when(col("stddev_transaction") / col("avg_transaction") > 2, 20.0).otherwise(0.0) +
        when(col("max_transaction") > 10000, 15.0).otherwise(0.0) +
        when(col("countries_visited") > 5, 10.0).otherwise(0.0)
      )
  }

  /**
   * Write results to S3 in Parquet format
   */
  def writeToS3(
    df: DataFrame,
    s3Path: String,
    mode: String = "overwrite",
    partitionBy: Seq[String] = Seq.empty
  ): Unit = {
    
    val writer = df.write
      .mode(mode)
      .option("compression", "snappy")
    
    if (partitionBy.nonEmpty) {
      writer.partitionBy(partitionBy: _*)
    }
    
    writer.parquet(s3Path)
  }

  /**
   * Write results back to PostgreSQL
   */
  def writeToPostgreSQL(
    df: DataFrame,
    tableName: String,
    mode: String = "append"
  ): Unit = {
    
    val jdbcUrl = sys.env.getOrElse("DB_URL", "jdbc:postgresql://localhost:5432/frauddb")
    val connectionProperties = new Properties()
    connectionProperties.put("user", sys.env.getOrElse("DB_USER", "postgres"))
    connectionProperties.put("password", sys.env.getOrElse("DB_PASSWORD", "postgres"))
    connectionProperties.put("driver", "org.postgresql.Driver")
    
    df.write
      .mode(mode)
      .jdbc(jdbcUrl, tableName, connectionProperties)
  }

  /**
   * Main entry point for Spark job
   * Can be submitted to EMR cluster
   */
  def main(args: Array[String]): Unit = {
    
    // Parse command line arguments
    val inputPath = if (args.length > 0) args(0) else "s3://bank-fraud-data/transactions/"
    val outputPath = if (args.length > 1) args(1) else "s3://bank-fraud-data/fraud-alerts/"
    val inputFormat = if (args.length > 2) args(2) else "parquet"
    
    println(s"Starting Spark Fraud Detection Job")
    println(s"Input: $inputPath")
    println(s"Output: $outputPath")
    println(s"Format: $inputFormat")
    
    // Create Spark session
    val spark = createSparkSession("BankFraudDetection-Batch")
    
    try {
      // Read transactions
      println("Reading transactions...")
      val transactions = readTransactions(spark, inputPath, inputFormat)
      println(s"Loaded ${transactions.count()} transactions")
      
      // Run fraud detection
      println("Running fraud detection pipeline...")
      val fraudAlerts = runFraudDetectionPipeline(transactions)
      println(s"Detected ${fraudAlerts.count()} fraud alerts")
      
      // Show sample results
      fraudAlerts.show(20, truncate = false)
      
      // Write results to S3
      println(s"Writing results to $outputPath")
      writeToS3(fraudAlerts, outputPath, partitionBy = Seq("risk_level"))
      
      // Generate customer risk profiles
      println("Generating customer risk profiles...")
      val customerProfiles = generateCustomerRiskProfiles(transactions)
      writeToS3(
        customerProfiles, 
        s"$outputPath/customer_profiles/",
        partitionBy = Seq.empty
      )
      
      // Write high-risk alerts back to PostgreSQL
      println("Writing high-risk alerts to PostgreSQL...")
      val highRiskAlerts = fraudAlerts.filter(col("risk_level").isin("HIGH", "CRITICAL"))
      writeToPostgreSQL(highRiskAlerts, "spark_fraud_alerts")
      
      println("Job completed successfully!")
      
    } catch {
      case e: Exception =>
        println(s"Job failed with error: ${e.getMessage}")
        e.printStackTrace()
        sys.exit(1)
    } finally {
      spark.stop()
    }
  }
}
