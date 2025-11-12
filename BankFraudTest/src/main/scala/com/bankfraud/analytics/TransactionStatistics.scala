package com.bankfraud.analytics

import scala.collection.mutable

/**
 * Advanced transaction statistics calculator using Scala.
 * Demonstrates functional programming and collection operations.
 * 
 * @author Banking Platform Team
 * @version 1.0
 */
object TransactionStatistics {
  
  /**
   * Calculates comprehensive statistics for a list of transaction amounts.
   * 
   * @param amounts list of transaction amounts
   * @return statistics map
   */
  def calculateStatistics(amounts: List[Double]): Map[String, Double] = {
    
    if (amounts.isEmpty) {
      return Map(
        "count" -> 0.0,
        "sum" -> 0.0,
        "mean" -> 0.0,
        "median" -> 0.0,
        "min" -> 0.0,
        "max" -> 0.0,
        "stdDev" -> 0.0
      )
    }
    
    val sorted = amounts.sorted
    val count = amounts.length
    val sum = amounts.sum
    val mean = sum / count
    
    val median = if (count % 2 == 0) {
      (sorted(count / 2 - 1) + sorted(count / 2)) / 2.0
    } else {
      sorted(count / 2)
    }
    
    val variance = amounts.map(a => math.pow(a - mean, 2)).sum / count
    val stdDev = math.sqrt(variance)
    
    Map(
      "count" -> count.toDouble,
      "sum" -> sum,
      "mean" -> mean,
      "median" -> median,
      "min" -> sorted.head,
      "max" -> sorted.last,
      "stdDev" -> stdDev,
      "variance" -> variance
    )
  }
  
  /**
   * Calculates percentiles for transaction amounts.
   * 
   * @param amounts list of transaction amounts
   * @param percentiles list of percentiles to calculate (e.g., 25, 50, 75, 95, 99)
   * @return map of percentile to value
   */
  def calculatePercentiles(
    amounts: List[Double],
    percentiles: List[Int] = List(25, 50, 75, 95, 99)
  ): Map[Int, Double] = {
    
    if (amounts.isEmpty) return Map.empty
    
    val sorted = amounts.sorted
    
    percentiles.map { p =>
      val index = math.ceil((p / 100.0) * sorted.length).toInt - 1
      val boundedIndex = math.max(0, math.min(index, sorted.length - 1))
      p -> sorted(boundedIndex)
    }.toMap
  }
  
  /**
   * Detects outliers using IQR (Interquartile Range) method.
   * 
   * @param amounts list of transaction amounts
   * @return list of outlier values
   */
  def detectOutliers(amounts: List[Double]): List[Double] = {
    
    if (amounts.length < 4) return List.empty
    
    val sorted = amounts.sorted
    val q1 = sorted((sorted.length * 0.25).toInt)
    val q3 = sorted((sorted.length * 0.75).toInt)
    val iqr = q3 - q1
    
    val lowerBound = q1 - (1.5 * iqr)
    val upperBound = q3 + (1.5 * iqr)
    
    amounts.filter(a => a < lowerBound || a > upperBound)
  }
  
  /**
   * Calculates moving average for time series data.
   * 
   * @param values time series values
   * @param windowSize size of moving window
   * @return list of moving averages
   */
  def movingAverage(values: List[Double], windowSize: Int): List[Double] = {
    
    if (values.length < windowSize) return List.empty
    
    values.sliding(windowSize).map { window =>
      window.sum / windowSize
    }.toList
  }
  
  /**
   * Groups transactions by time bucket and calculates statistics.
   * 
   * @param timestamps list of timestamps
   * @param amounts corresponding amounts
   * @param bucketSize bucket size in hours
   * @return map of bucket to statistics
   */
  def groupByTimeBucket(
    timestamps: List[Long],
    amounts: List[Double],
    bucketSize: Int = 24
  ): Map[Long, Map[String, Double]] = {
    
    val millisPerHour = 3600000L
    val bucketMillis = bucketSize * millisPerHour
    
    val grouped = timestamps.zip(amounts).groupBy { case (timestamp, _) =>
      (timestamp / bucketMillis) * bucketMillis
    }
    
    grouped.mapValues { group =>
      val bucketAmounts = group.map(_._2)
      calculateStatistics(bucketAmounts)
    }.toMap
  }
  
  /**
   * Calculates trend (increasing/decreasing/stable).
   * 
   * @param values time series values
   * @return trend description
   */
  def calculateTrend(values: List[Double]): String = {
    
    if (values.length < 2) return "INSUFFICIENT_DATA"
    
    val differences = values.sliding(2).map { case List(a, b) => b - a }.toList
    val avgDiff = differences.sum / differences.length
    
    val threshold = values.sum / values.length * 0.05 // 5% threshold
    
    if (avgDiff > threshold) "INCREASING"
    else if (avgDiff < -threshold) "DECREASING"
    else "STABLE"
  }
  
  /**
   * Calculates correlation coefficient between two series.
   * 
   * @param series1 first data series
   * @param series2 second data series
   * @return correlation coefficient (-1 to 1)
   */
  def correlation(series1: List[Double], series2: List[Double]): Double = {
    
    if (series1.length != series2.length || series1.isEmpty) return 0.0
    
    val mean1 = series1.sum / series1.length
    val mean2 = series2.sum / series2.length
    
    val covariance = series1.zip(series2).map { case (x, y) =>
      (x - mean1) * (y - mean2)
    }.sum / series1.length
    
    val stdDev1 = math.sqrt(series1.map(x => math.pow(x - mean1, 2)).sum / series1.length)
    val stdDev2 = math.sqrt(series2.map(y => math.pow(y - mean2, 2)).sum / series2.length)
    
    if (stdDev1 == 0 || stdDev2 == 0) 0.0
    else covariance / (stdDev1 * stdDev2)
  }
}
