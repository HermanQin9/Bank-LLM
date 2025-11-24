#!/bin/bash
# AWS EMR Bootstrap Script
# Installs additional dependencies and configures EMR cluster

set -e

echo "=========================================="
echo "EMR Bootstrap Script - Bank Fraud Detection"
echo "=========================================="

# Update system packages
echo "Updating system packages..."
sudo yum update -y

# Install PostgreSQL client
echo "Installing PostgreSQL client..."
sudo yum install -y postgresql postgresql-devel

# Install Python dependencies for PySpark
echo "Installing Python packages..."
sudo pip3 install --upgrade pip
sudo pip3 install \
    pandas \
    numpy \
    psycopg2-binary \
    boto3 \
    pyarrow \
    fastparquet

# Configure Spark defaults
echo "Configuring Spark defaults..."
cat << 'EOF' | sudo tee -a /etc/spark/conf/spark-defaults.conf
# Performance optimizations
spark.executor.memory                   4g
spark.executor.cores                    2
spark.driver.memory                     2g
spark.sql.adaptive.enabled              true
spark.sql.adaptive.coalescePartitions.enabled true
spark.serializer                        org.apache.spark.serializer.KryoSerializer
spark.kryoserializer.buffer.max         512m

# S3 optimizations
spark.hadoop.fs.s3a.connection.maximum  100
spark.hadoop.fs.s3a.threads.max         256
spark.hadoop.fs.s3a.fast.upload         true
spark.hadoop.fs.s3a.multipart.size      104857600

# Dynamic allocation
spark.dynamicAllocation.enabled         true
spark.dynamicAllocation.minExecutors    1
spark.dynamicAllocation.maxExecutors    10
spark.shuffle.service.enabled           true

# Monitoring
spark.eventLog.enabled                  true
spark.history.fs.logDirectory           s3://bank-fraud-data/spark-logs/
EOF

# Download and configure PostgreSQL JDBC driver
echo "Downloading PostgreSQL JDBC driver..."
sudo wget -P /usr/lib/spark/jars/ \
    https://jdbc.postgresql.org/download/postgresql-42.7.1.jar

# Create application directories
echo "Creating application directories..."
mkdir -p /mnt/spark-work
mkdir -p /mnt/spark-tmp
sudo chmod 777 /mnt/spark-work
sudo chmod 777 /mnt/spark-tmp

# Set environment variables
echo "Setting environment variables..."
cat << 'EOF' | sudo tee -a /etc/environment
SPARK_HOME=/usr/lib/spark
SPARK_WORK_DIR=/mnt/spark-work
SPARK_LOCAL_DIRS=/mnt/spark-tmp
EOF

# Configure log4j for better logging
echo "Configuring Spark logging..."
cat << 'EOF' | sudo tee /etc/spark/conf/log4j.properties
log4j.rootCategory=INFO, console
log4j.appender.console=org.apache.log4j.ConsoleAppender
log4j.appender.console.target=System.err
log4j.appender.console.layout=org.apache.log4j.PatternLayout
log4j.appender.console.layout.ConversionPattern=%d{yy/MM/dd HH:mm:ss} %p %c{1}: %m%n

# Reduce logging verbosity
log4j.logger.org.apache.spark.repl.Main=WARN
log4j.logger.org.spark_project.jetty=WARN
log4j.logger.org.spark_project.jetty.util.component.AbstractLifeCycle=ERROR
log4j.logger.org.apache.spark.repl.SparkIMain$exprTyper=INFO
log4j.logger.org.apache.spark.repl.SparkILoop$SparkILoopInterpreter=INFO
log4j.logger.org.apache.parquet=ERROR
log4j.logger.parquet=ERROR
EOF

# Download application JAR from S3 (if available)
if [ -n "$APPLICATION_JAR_S3" ]; then
    echo "Downloading application JAR from S3..."
    aws s3 cp "$APPLICATION_JAR_S3" /home/hadoop/fraud-detection.jar
fi

echo "=========================================="
echo "Bootstrap completed successfully!"
echo "=========================================="
