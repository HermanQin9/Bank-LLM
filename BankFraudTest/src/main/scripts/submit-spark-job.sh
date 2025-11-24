#!/bin/bash
# Submit Spark job to AWS EMR cluster
# Usage: ./submit-spark-job.sh [cluster-id] [job-type] [input-path] [output-path]

set -e

# Configuration
CLUSTER_ID=${1:-""}
JOB_TYPE=${2:-"batch"}  # batch or streaming
INPUT_PATH=${3:-"s3://bank-fraud-data/transactions/"}
OUTPUT_PATH=${4:-"s3://bank-fraud-data/fraud-alerts/"}
JAR_PATH="s3://bank-fraud-data/jars/banking-platform-migration-1.0.0.jar"
LOG_URI="s3://bank-fraud-data/spark-logs/"

if [ -z "$CLUSTER_ID" ]; then
    echo "Error: Cluster ID is required"
    echo "Usage: $0 <cluster-id> [job-type] [input-path] [output-path]"
    exit 1
fi

echo "=========================================="
echo "Submitting Spark Job to EMR"
echo "=========================================="
echo "Cluster ID: $CLUSTER_ID"
echo "Job Type: $JOB_TYPE"
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "=========================================="

if [ "$JOB_TYPE" == "batch" ]; then
    # Submit batch processing job
    STEP_ID=$(aws emr add-steps \
        --cluster-id "$CLUSTER_ID" \
        --steps Type=Spark,Name="Fraud Detection Batch",\
ActionOnFailure=CONTINUE,\
Args=[--class,com.bankfraud.spark.SparkTransactionProcessor,\
--master,yarn,\
--deploy-mode,cluster,\
--driver-memory,2g,\
--executor-memory,4g,\
--executor-cores,2,\
--num-executors,5,\
--conf,spark.sql.adaptive.enabled=true,\
--conf,spark.dynamicAllocation.enabled=true,\
$JAR_PATH,\
$INPUT_PATH,\
$OUTPUT_PATH,\
parquet] \
        --query 'StepIds[0]' \
        --output text)

elif [ "$JOB_TYPE" == "streaming" ]; then
    # Submit streaming job
    KAFKA_SERVERS=${5:-"localhost:9092"}
    KAFKA_TOPIC=${6:-"transactions"}
    
    STEP_ID=$(aws emr add-steps \
        --cluster-id "$CLUSTER_ID" \
        --steps Type=Spark,Name="Fraud Detection Streaming",\
ActionOnFailure=CONTINUE,\
Args=[--class,com.bankfraud.spark.SparkStreamingProcessor,\
--master,yarn,\
--deploy-mode,cluster,\
--driver-memory,2g,\
--executor-memory,4g,\
--executor-cores,2,\
--conf,spark.streaming.stopGracefullyOnShutdown=true,\
--packages,org.apache.spark:spark-sql-kafka-0-10_2.13:3.4.0,\
$JAR_PATH,\
kafka,\
$KAFKA_SERVERS,\
$KAFKA_TOPIC,\
s3,\
$OUTPUT_PATH] \
        --query 'StepIds[0]' \
        --output text)
else
    echo "Error: Invalid job type. Use 'batch' or 'streaming'"
    exit 1
fi

echo "Step ID: $STEP_ID"
echo "Monitoring job progress..."
echo "=========================================="

# Monitor step status
while true; do
    STATUS=$(aws emr describe-step \
        --cluster-id "$CLUSTER_ID" \
        --step-id "$STEP_ID" \
        --query 'Step.Status.State' \
        --output text)
    
    echo "Status: $STATUS ($(date))"
    
    case $STATUS in
        COMPLETED)
            echo "=========================================="
            echo "Job completed successfully!"
            echo "=========================================="
            echo "Output location: $OUTPUT_PATH"
            exit 0
            ;;
        FAILED|CANCELLED|INTERRUPTED)
            echo "=========================================="
            echo "Job failed with status: $STATUS"
            echo "=========================================="
            echo "Check logs at: $LOG_URI"
            exit 1
            ;;
        PENDING|RUNNING)
            sleep 30
            ;;
    esac
done
