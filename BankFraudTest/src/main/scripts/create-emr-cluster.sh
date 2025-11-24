#!/bin/bash
# Create AWS EMR cluster with custom configuration
# Usage: ./create-emr-cluster.sh

set -e

echo "=========================================="
echo "Creating AWS EMR Cluster"
echo "=========================================="

# Upload bootstrap script to S3
echo "Uploading bootstrap script to S3..."
aws s3 cp emr-bootstrap.sh s3://bank-fraud-data/scripts/emr-bootstrap.sh

# Upload application JAR to S3
echo "Uploading application JAR to S3..."
cd ../../..
mvn clean package -DskipTests
aws s3 cp target/banking-platform-migration-1.0.0.jar s3://bank-fraud-data/jars/

cd src/main/scripts

# Create EMR cluster
echo "Creating EMR cluster..."
CLUSTER_ID=$(aws emr create-cluster \
    --cli-input-json file://emr-cluster-config.json \
    --query 'ClusterId' \
    --output text)

echo "=========================================="
echo "EMR Cluster Created Successfully!"
echo "=========================================="
echo "Cluster ID: $CLUSTER_ID"
echo ""
echo "Monitoring cluster status..."
echo "=========================================="

# Wait for cluster to be ready
while true; do
    STATUS=$(aws emr describe-cluster \
        --cluster-id "$CLUSTER_ID" \
        --query 'Cluster.Status.State' \
        --output text)
    
    echo "Cluster Status: $STATUS ($(date))"
    
    case $STATUS in
        WAITING)
            echo "=========================================="
            echo "Cluster is ready and waiting for jobs!"
            echo "=========================================="
            echo ""
            echo "To submit a batch job:"
            echo "  ./submit-spark-job.sh $CLUSTER_ID batch"
            echo ""
            echo "To submit a streaming job:"
            echo "  ./submit-spark-job.sh $CLUSTER_ID streaming"
            echo ""
            echo "To terminate cluster:"
            echo "  aws emr terminate-clusters --cluster-ids $CLUSTER_ID"
            echo ""
            echo "Master node public DNS:"
            aws emr describe-cluster \
                --cluster-id "$CLUSTER_ID" \
                --query 'Cluster.MasterPublicDnsName' \
                --output text
            exit 0
            ;;
        TERMINATED|TERMINATED_WITH_ERRORS)
            echo "=========================================="
            echo "Cluster creation failed!"
            echo "Status: $STATUS"
            echo "=========================================="
            exit 1
            ;;
        STARTING|BOOTSTRAPPING|RUNNING)
            sleep 30
            ;;
    esac
done
