# Multi-stage build for Java Transaction Processing Service
# Stage 1: Build with Maven
FROM maven:3.9-eclipse-temurin-21-alpine AS build

LABEL maintainer="Herman Qin <hermantqin@gmail.com>"
LABEL description="Financial Transaction Processing Service with Scala Fraud Detection"

WORKDIR /build

# Copy pom.xml and download dependencies (cached layer)
COPY pom.xml .
RUN mvn dependency:go-offline -B

# Copy source code
COPY src ./src

# Build application (skip tests in Docker build, run in CI/CD)
RUN mvn clean package -DskipTests -B

# Verify JAR was created
RUN ls -lh target/*.jar

# Stage 2: Production runtime
FROM eclipse-temurin:21-jre-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    postgresql-client \
    && rm -rf /var/cache/apk/*

# Create app user for security (don't run as root)
RUN addgroup -g 1001 appuser && \
    adduser -D -u 1001 -G appuser appuser

WORKDIR /app

# Copy JAR from build stage
COPY --from=build /build/target/*.jar app.jar

# Copy data and migration scripts
COPY data ./data
COPY src/main/resources/db/migration ./db/migration

# Create necessary directories
RUN mkdir -p logs data/processed && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# JVM options for production
ENV JAVA_OPTS="-Xms512m -Xmx2g \
    -XX:+UseG1GC \
    -XX:MaxGCPauseMillis=200 \
    -XX:+HeapDumpOnOutOfMemoryError \
    -XX:HeapDumpPath=/app/logs/heap_dump.hprof \
    -Djava.security.egd=file:/dev/./urandom \
    -Dfile.encoding=UTF-8"

# Application properties
ENV SERVER_PORT=8080 \
    SPRING_PROFILES_ACTIVE=production

# Entry point with JVM options
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]

# Metadata
LABEL org.opencontainers.image.source="https://github.com/HermanQin9/Bank-LLM"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.title="Financial Transaction Service"
