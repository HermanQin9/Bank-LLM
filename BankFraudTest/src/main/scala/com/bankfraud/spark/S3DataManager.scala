package com.bankfraud.spark

import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import software.amazon.awssdk.services.s3.S3Client
import software.amazon.awssdk.services.s3.model._
import software.amazon.awssdk.regions.Region
import scala.collection.JavaConverters._

/**
 * AWS S3 Integration for Spark Data Processing
 * 
 * This module provides:
 * - Optimized S3 read/write operations
 * - Parquet format support for efficient storage
 * - Partition management for large datasets
 * - S3 file listing and metadata operations
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
object S3DataManager {

  /**
   * Configure SparkSession with S3 optimizations
   */
  def createS3OptimizedSparkSession(appName: String = "BankFraud-S3"): SparkSession = {
    SparkSession.builder()
      .appName(appName)
      // S3A configuration for AWS
      .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
      .config("spark.hadoop.fs.s3a.aws.credentials.provider", 
        "com.amazonaws.auth.DefaultAWSCredentialsProviderChain")
      .config("spark.hadoop.fs.s3a.connection.maximum", "100")
      .config("spark.hadoop.fs.s3a.threads.max", "256")
      .config("spark.hadoop.fs.s3a.fast.upload", "true")
      .config("spark.hadoop.fs.s3a.multipart.size", "104857600") // 100MB
      .config("spark.hadoop.fs.s3a.multipart.threshold", "2147483647") // 2GB
      // Parquet optimizations
      .config("spark.sql.parquet.compression.codec", "snappy")
      .config("spark.sql.parquet.mergeSchema", "false")
      .config("spark.sql.parquet.filterPushdown", "true")
      .config("spark.sql.parquet.enableVectorizedReader", "true")
      // Performance
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()
  }

  /**
   * Read data from S3 in Parquet format
   * 
   * @param spark SparkSession
   * @param s3Path S3 path (e.g., s3://bucket/path/)
   * @param partitionColumns Optional partition columns to filter
   * @return DataFrame
   */
  def readFromS3Parquet(
    spark: SparkSession,
    s3Path: String,
    partitionColumns: Map[String, String] = Map.empty
  ): DataFrame = {
    
    println(s"Reading from S3: $s3Path")
    
    var reader = spark.read
      .format("parquet")
      .option("mergeSchema", "false")
    
    // Apply partition filters if provided
    if (partitionColumns.nonEmpty) {
      val filterExpr = partitionColumns.map { 
        case (col, value) => s"$col = '$value'" 
      }.mkString(" AND ")
      reader = reader.option("pathGlobFilter", s"*{$filterExpr}*")
    }
    
    val df = reader.load(s3Path)
    println(s"Loaded ${df.count()} records from S3")
    df
  }

  /**
   * Read data from S3 in CSV format
   */
  def readFromS3CSV(
    spark: SparkSession,
    s3Path: String,
    header: Boolean = true,
    inferSchema: Boolean = false
  ): DataFrame = {
    
    spark.read
      .format("csv")
      .option("header", header.toString)
      .option("inferSchema", inferSchema.toString)
      .option("dateFormat", "yyyy-MM-dd HH:mm:ss")
      .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
      .load(s3Path)
  }

  /**
   * Read data from S3 in JSON format
   */
  def readFromS3JSON(
    spark: SparkSession,
    s3Path: String,
    multiLine: Boolean = false
  ): DataFrame = {
    
    spark.read
      .format("json")
      .option("multiLine", multiLine.toString)
      .option("dateFormat", "yyyy-MM-dd'T'HH:mm:ss.SSSZ")
      .load(s3Path)
  }

  /**
   * Write DataFrame to S3 in Parquet format with optimizations
   * 
   * @param df DataFrame to write
   * @param s3Path S3 destination path
   * @param mode Save mode (overwrite, append, ignore, error)
   * @param partitionBy Columns to partition by
   * @param coalesce Number of output files (optional)
   */
  def writeToS3Parquet(
    df: DataFrame,
    s3Path: String,
    mode: SaveMode = SaveMode.Overwrite,
    partitionBy: Seq[String] = Seq.empty,
    coalesce: Option[Int] = None
  ): Unit = {
    
    println(s"Writing ${df.count()} records to S3: $s3Path")
    
    var writer = df.write
      .mode(mode)
      .format("parquet")
      .option("compression", "snappy")
      .option("maxRecordsPerFile", "1000000") // 1M records per file
    
    // Coalesce to reduce number of files
    val dataToWrite = coalesce match {
      case Some(n) => df.coalesce(n)
      case None => df
    }
    
    writer = dataToWrite.write.mode(mode).format("parquet")
    
    // Add partitioning if specified
    if (partitionBy.nonEmpty) {
      writer = writer.partitionBy(partitionBy: _*)
    }
    
    writer.save(s3Path)
    println(s"Successfully wrote data to $s3Path")
  }

  /**
   * Write DataFrame to S3 in CSV format
   */
  def writeToS3CSV(
    df: DataFrame,
    s3Path: String,
    mode: SaveMode = SaveMode.Overwrite,
    header: Boolean = true
  ): Unit = {
    
    df.write
      .mode(mode)
      .format("csv")
      .option("header", header.toString)
      .option("dateFormat", "yyyy-MM-dd HH:mm:ss")
      .save(s3Path)
  }

  /**
   * List files in S3 bucket/prefix
   */
  def listS3Files(bucketName: String, prefix: String, region: String = "us-east-1"): Seq[String] = {
    
    val s3Client = S3Client.builder()
      .region(Region.of(region))
      .build()
    
    try {
      val request = ListObjectsV2Request.builder()
        .bucket(bucketName)
        .prefix(prefix)
        .build()
      
      val response = s3Client.listObjectsV2(request)
      
      response.contents().asScala.map(_.key()).toSeq
      
    } finally {
      s3Client.close()
    }
  }

  /**
   * Get S3 file metadata (size, last modified)
   */
  def getS3FileMetadata(
    bucketName: String, 
    key: String, 
    region: String = "us-east-1"
  ): Option[Map[String, Any]] = {
    
    val s3Client = S3Client.builder()
      .region(Region.of(region))
      .build()
    
    try {
      val request = HeadObjectRequest.builder()
        .bucket(bucketName)
        .key(key)
        .build()
      
      val response = s3Client.headObject(request)
      
      Some(Map(
        "size" -> response.contentLength(),
        "lastModified" -> response.lastModified().toString,
        "contentType" -> response.contentType()
      ))
      
    } catch {
      case _: Exception => None
    } finally {
      s3Client.close()
    }
  }

  /**
   * Delete files from S3
   */
  def deleteS3Files(
    bucketName: String, 
    keys: Seq[String], 
    region: String = "us-east-1"
  ): Unit = {
    
    val s3Client = S3Client.builder()
      .region(Region.of(region))
      .build()
    
    try {
      val objectIds = keys.map { key =>
        ObjectIdentifier.builder().key(key).build()
      }.asJava
      
      val deleteRequest = Delete.builder()
        .objects(objectIds)
        .build()
      
      val request = DeleteObjectsRequest.builder()
        .bucket(bucketName)
        .delete(deleteRequest)
        .build()
      
      s3Client.deleteObjects(request)
      println(s"Deleted ${keys.length} files from S3")
      
    } finally {
      s3Client.close()
    }
  }

  /**
   * Copy transactions to S3 for processing
   * Useful for migrating data from PostgreSQL to S3 data lake
   */
  def exportTransactionsToS3(
    spark: SparkSession,
    jdbcUrl: String,
    s3OutputPath: String,
    dateFrom: Option[String] = None,
    dateTo: Option[String] = None
  ): Unit = {
    
    println("Exporting transactions from PostgreSQL to S3...")
    
    // Build query with optional date filters
    val baseQuery = "SELECT * FROM transactions"
    val dateFilter = (dateFrom, dateTo) match {
      case (Some(from), Some(to)) => 
        s" WHERE transaction_date >= '$from' AND transaction_date < '$to'"
      case (Some(from), None) => 
        s" WHERE transaction_date >= '$from'"
      case (None, Some(to)) => 
        s" WHERE transaction_date < '$to'"
      case _ => ""
    }
    
    val query = s"($baseQuery$dateFilter) as transactions"
    
    // Read from PostgreSQL
    val transactions = spark.read
      .format("jdbc")
      .option("url", jdbcUrl)
      .option("dbtable", query)
      .option("user", sys.env.getOrElse("DB_USER", "postgres"))
      .option("password", sys.env.getOrElse("DB_PASSWORD", "postgres"))
      .option("driver", "org.postgresql.Driver")
      .option("fetchsize", "10000")
      .load()
    
    println(s"Loaded ${transactions.count()} transactions from PostgreSQL")
    
    // Add processing date for partitioning
    val withPartitions = transactions
      .withColumn("year", year(col("transaction_date")))
      .withColumn("month", month(col("transaction_date")))
      .withColumn("day", dayofmonth(col("transaction_date")))
    
    // Write to S3 with partitioning
    writeToS3Parquet(
      withPartitions,
      s3OutputPath,
      mode = SaveMode.Append,
      partitionBy = Seq("year", "month", "day")
    )
    
    println("Export completed successfully!")
  }

  /**
   * Example: Process daily transaction batch from S3
   */
  def processDailyBatch(
    spark: SparkSession,
    bucketName: String,
    year: Int,
    month: Int,
    day: Int
  ): DataFrame = {
    
    val s3Path = f"s3a://$bucketName/transactions/year=$year/month=$month%02d/day=$day%02d/"
    
    println(s"Processing daily batch: $s3Path")
    
    val transactions = readFromS3Parquet(spark, s3Path)
    
    // Apply fraud detection
    val fraudAlerts = SparkTransactionProcessor.runFraudDetectionPipeline(transactions)
    
    // Write results back to S3
    val outputPath = f"s3a://$bucketName/fraud-alerts/year=$year/month=$month%02d/day=$day%02d/"
    writeToS3Parquet(fraudAlerts, outputPath)
    
    fraudAlerts
  }

  /**
   * Main entry point for S3 data operations
   */
  def main(args: Array[String]): Unit = {
    
    val operation = if (args.length > 0) args(0) else "export"
    val bucketName = sys.env.getOrElse("S3_BUCKET", "bank-fraud-data")
    
    val spark = createS3OptimizedSparkSession()
    
    try {
      operation match {
        case "export" =>
          // Export PostgreSQL to S3
          val jdbcUrl = sys.env.getOrElse("DB_URL", "jdbc:postgresql://localhost:5432/frauddb")
          val outputPath = s"s3a://$bucketName/transactions/"
          exportTransactionsToS3(spark, jdbcUrl, outputPath)
        
        case "process" =>
          // Process daily batch
          val year = if (args.length > 1) args(1).toInt else 2025
          val month = if (args.length > 2) args(2).toInt else 11
          val day = if (args.length > 3) args(3).toInt else 24
          processDailyBatch(spark, bucketName, year, month, day)
        
        case "list" =>
          // List S3 files
          val prefix = if (args.length > 1) args(1) else "transactions/"
          val files = listS3Files(bucketName, prefix)
          println(s"Found ${files.length} files:")
          files.take(20).foreach(println)
        
        case _ =>
          println(s"Unknown operation: $operation")
          println("Valid operations: export, process, list")
      }
      
    } catch {
      case e: Exception =>
        println(s"Operation failed: ${e.getMessage}")
        e.printStackTrace()
        sys.exit(1)
    } finally {
      spark.stop()
    }
  }
}
