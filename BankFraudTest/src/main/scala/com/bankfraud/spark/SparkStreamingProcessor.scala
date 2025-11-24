package com.bankfraud.spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.streaming.{OutputMode, Trigger}
import org.apache.spark.sql.types._
import java.util.Properties

/**
 * Apache Spark Structured Streaming for Real-Time Transaction Processing
 * 
 * This module provides:
 * - Real-time transaction ingestion from Kafka/Kinesis
 * - Streaming fraud detection with windowed aggregations
 * - Integration with existing FraudAnalyzer logic
 * - Low-latency alerting (<5 seconds)
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
object SparkStreamingProcessor {

  /**
   * Create SparkSession optimized for streaming workloads
   */
  def createStreamingSparkSession(appName: String = "BankFraudDetection-Streaming"): SparkSession = {
    SparkSession.builder()
      .appName(appName)
      .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoints")
      .config("spark.sql.shuffle.partitions", "10")
      .config("spark.streaming.stopGracefullyOnShutdown", "true")
      .config("spark.sql.streaming.schemaInference", "true")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()
  }

  /**
   * Read streaming data from Kafka
   */
  def readFromKafka(
    spark: SparkSession,
    kafkaBootstrapServers: String,
    topic: String
  ): DataFrame = {
    
    spark.readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("subscribe", topic)
      .option("startingOffsets", "latest")
      .option("failOnDataLoss", "false")
      .load()
      .selectExpr("CAST(value AS STRING) as json_data")
      .select(from_json(col("json_data"), SparkTransactionProcessor.transactionSchema).as("data"))
      .select("data.*")
  }

  /**
   * Read streaming data from AWS Kinesis
   */
  def readFromKinesis(
    spark: SparkSession,
    streamName: String,
    region: String = "us-east-1"
  ): DataFrame = {
    
    spark.readStream
      .format("kinesis")
      .option("streamName", streamName)
      .option("region", region)
      .option("initialPosition", "LATEST")
      .option("format", "json")
      .load()
      .select(from_json(col("data").cast("string"), SparkTransactionProcessor.transactionSchema).as("transaction"))
      .select("transaction.*")
  }

  /**
   * Real-time high-value transaction detection
   */
  def streamHighValueDetection(transactions: DataFrame): DataFrame = {
    transactions
      .filter(col("amount") > 5000.0)
      .withColumn("risk_factor", lit("HIGH_VALUE"))
      .withColumn("risk_score", lit(25.0))
      .withColumn("alert_timestamp", current_timestamp())
  }

  /**
   * Real-time velocity detection using windowed aggregations
   * Detects multiple transactions within sliding time window
   */
  def streamVelocityDetection(
    transactions: DataFrame,
    windowDuration: String = "1 hour",
    slideDuration: String = "10 minutes"
  ): DataFrame = {
    
    transactions
      .withWatermark("transaction_date", "30 minutes")
      .groupBy(
        col("customer_id"),
        window(col("transaction_date"), windowDuration, slideDuration)
      )
      .agg(
        count("*").as("transaction_count"),
        sum("amount").as("total_amount"),
        collect_list("transaction_id").as("transaction_ids")
      )
      .filter(col("transaction_count") >= 3)
      .select(
        col("customer_id"),
        col("window.start").as("window_start"),
        col("window.end").as("window_end"),
        col("transaction_count"),
        col("total_amount"),
        col("transaction_ids")
      )
      .withColumn("risk_factor", lit("HIGH_VELOCITY"))
      .withColumn("risk_score",
        when(col("transaction_count") >= 5, 30.0)
        .when(col("transaction_count") >= 3, 20.0)
        .otherwise(10.0)
      )
      .withColumn("alert_timestamp", current_timestamp())
  }

  /**
   * Stateful streaming for customer behavior tracking
   * Maintains running statistics per customer
   */
  def streamStatisticalAnomalies(transactions: DataFrame): DataFrame = {
    
    // Running aggregations per customer
    val customerStats = transactions
      .groupBy("customer_id")
      .agg(
        avg("amount").as("running_avg"),
        stddev("amount").as("running_stddev"),
        count("*").as("running_count")
      )
    
    // Join incoming transactions with running stats
    transactions
      .join(customerStats, "customer_id")
      .withColumn("z_score",
        when(col("running_stddev") > 0,
          abs(col("amount") - col("running_avg")) / col("running_stddev")
        ).otherwise(0.0)
      )
      .filter(col("z_score") >= 2.0)
      .withColumn("risk_factor", lit("STATISTICAL_ANOMALY"))
      .withColumn("risk_score",
        when(col("z_score") >= 3, 25.0)
        .when(col("z_score") >= 2, 15.0)
        .otherwise(10.0)
      )
      .withColumn("alert_timestamp", current_timestamp())
  }

  /**
   * Real-time geographic anomaly detection
   * Detects impossible travel patterns
   */
  def streamGeographicAnomalies(transactions: DataFrame): DataFrame = {
    
    import org.apache.spark.sql.expressions.Window
    
    val windowSpec = Window
      .partitionBy("customer_id")
      .orderBy(col("transaction_date").cast("long"))
    
    transactions
      .withWatermark("transaction_date", "1 hour")
      .withColumn("prev_country", lag("location_country", 1).over(windowSpec))
      .withColumn("prev_time", lag("transaction_date", 1).over(windowSpec))
      .withColumn("time_diff_hours",
        (unix_timestamp(col("transaction_date")) - unix_timestamp(col("prev_time"))) / 3600.0
      )
      .filter(
        col("prev_country").isNotNull &&
        col("location_country") =!= col("prev_country") &&
        col("time_diff_hours") < 6.0
      )
      .withColumn("risk_factor", lit("GEOGRAPHIC_ANOMALY"))
      .withColumn("risk_score", lit(20.0))
      .withColumn("alert_timestamp", current_timestamp())
  }

  /**
   * Write streaming results to Kafka for downstream consumption
   */
  def writeToKafka(
    streamingDF: DataFrame,
    kafkaBootstrapServers: String,
    outputTopic: String,
    checkpointPath: String
  ): Unit = {
    
    streamingDF
      .selectExpr("to_json(struct(*)) AS value")
      .writeStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("topic", outputTopic)
      .option("checkpointLocation", checkpointPath)
      .outputMode(OutputMode.Append())
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .start()
      .awaitTermination()
  }

  /**
   * Write streaming results to PostgreSQL
   * Uses foreachBatch for efficient batch writes
   */
  def writeToPostgreSQL(
    streamingDF: DataFrame,
    tableName: String,
    checkpointPath: String
  ): Unit = {
    
    val jdbcUrl = sys.env.getOrElse("DB_URL", "jdbc:postgresql://localhost:5432/frauddb")
    val dbUser = sys.env.getOrElse("DB_USER", "postgres")
    val dbPassword = sys.env.getOrElse("DB_PASSWORD", "postgres")
    
    streamingDF.writeStream
      .foreachBatch { (batchDF: DataFrame, batchId: Long) =>
        println(s"Processing batch $batchId with ${batchDF.count()} records")
        
        val connectionProperties = new Properties()
        connectionProperties.put("user", dbUser)
        connectionProperties.put("password", dbPassword)
        connectionProperties.put("driver", "org.postgresql.Driver")
        
        batchDF.write
          .mode("append")
          .jdbc(jdbcUrl, tableName, connectionProperties)
      }
      .option("checkpointLocation", checkpointPath)
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start()
      .awaitTermination()
  }

  /**
   * Write streaming results to S3 with partitioning
   */
  def writeToS3(
    streamingDF: DataFrame,
    s3Path: String,
    checkpointPath: String,
    partitionBy: Seq[String] = Seq("risk_level")
  ): Unit = {
    
    val writer = streamingDF.writeStream
      .format("parquet")
      .option("path", s3Path)
      .option("checkpointLocation", checkpointPath)
      .outputMode(OutputMode.Append())
      .trigger(Trigger.ProcessingTime("1 minute"))
    
    if (partitionBy.nonEmpty) {
      writer.partitionBy(partitionBy: _*)
    }
    
    writer.start().awaitTermination()
  }

  /**
   * Console output for debugging (development only)
   */
  def writeToConsole(streamingDF: DataFrame): Unit = {
    streamingDF.writeStream
      .format("console")
      .outputMode(OutputMode.Append())
      .trigger(Trigger.ProcessingTime("5 seconds"))
      .option("truncate", false)
      .start()
      .awaitTermination()
  }

  /**
   * Main entry point for streaming job
   */
  def main(args: Array[String]): Unit = {
    
    // Parse arguments
    val inputSource = if (args.length > 0) args(0) else "kafka"
    val inputEndpoint = if (args.length > 1) args(1) else "localhost:9092"
    val inputTopic = if (args.length > 2) args(2) else "transactions"
    val outputSink = if (args.length > 3) args(3) else "console"
    val outputPath = if (args.length > 4) args(4) else "s3://bank-fraud-data/streaming-alerts/"
    
    println(s"Starting Spark Streaming Fraud Detection")
    println(s"Input Source: $inputSource")
    println(s"Input Endpoint: $inputEndpoint")
    println(s"Input Topic: $inputTopic")
    println(s"Output Sink: $outputSink")
    
    val spark = createStreamingSparkSession()
    
    try {
      // Read streaming transactions
      val transactions = inputSource.toLowerCase match {
        case "kafka" => readFromKafka(spark, inputEndpoint, inputTopic)
        case "kinesis" => readFromKinesis(spark, inputTopic)
        case _ => throw new IllegalArgumentException(s"Unsupported input source: $inputSource")
      }
      
      println("Starting streaming queries...")
      
      // Apply fraud detection
      val highValueAlerts = streamHighValueDetection(transactions)
      val velocityAlerts = streamVelocityDetection(transactions)
      val statisticalAlerts = streamStatisticalAnomalies(transactions)
      val geoAlerts = streamGeographicAnomalies(transactions)
      
      // Union all alerts
      val allAlerts = highValueAlerts
        .union(velocityAlerts.drop("window_start", "window_end", "transaction_count", "total_amount", "transaction_ids"))
        .union(statisticalAlerts)
        .union(geoAlerts)
      
      // Add risk level classification
      val classifiedAlerts = allAlerts
        .withColumn("risk_level",
          when(col("risk_score") >= 25, "HIGH")
          .when(col("risk_score") >= 15, "MEDIUM")
          .otherwise("LOW")
        )
      
      // Write to output sink
      outputSink.toLowerCase match {
        case "console" =>
          writeToConsole(classifiedAlerts)
        
        case "kafka" =>
          val outputTopic = if (args.length > 5) args(5) else "fraud-alerts"
          writeToKafka(classifiedAlerts, inputEndpoint, outputTopic, s"$outputPath/checkpoints/kafka")
        
        case "postgresql" =>
          writeToPostgreSQL(classifiedAlerts, "streaming_fraud_alerts", s"$outputPath/checkpoints/postgres")
        
        case "s3" =>
          writeToS3(classifiedAlerts, outputPath, s"$outputPath/checkpoints/s3")
        
        case _ =>
          throw new IllegalArgumentException(s"Unsupported output sink: $outputSink")
      }
      
    } catch {
      case e: Exception =>
        println(s"Streaming job failed: ${e.getMessage}")
        e.printStackTrace()
        sys.exit(1)
    } finally {
      spark.stop()
    }
  }
}
