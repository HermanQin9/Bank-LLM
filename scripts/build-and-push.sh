#!/bin/bash

# Docker Build and Push Script
# Builds and pushes Docker images to Docker Hub and AWS ECR

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-hermantqin}"
VERSION="${VERSION:-latest}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Docker Images${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Version: $VERSION"
echo "Git Commit: $GIT_COMMIT"
echo "Build Date: $BUILD_DATE"
echo ""

# Build Java Service
echo -e "${YELLOW}Building Java Transaction Service...${NC}"
docker build \
    -t ${DOCKER_USERNAME}/financial-java-service:${VERSION} \
    -t ${DOCKER_USERNAME}/financial-java-service:${GIT_COMMIT} \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg GIT_COMMIT="${GIT_COMMIT}" \
    --platform linux/amd64 \
    -f BankFraudTest/Dockerfile \
    ./BankFraudTest

echo -e "${GREEN}✅ Java service image built${NC}"

# Build Python API
echo -e "${YELLOW}Building Python ML/LLM API...${NC}"
docker build \
    -t ${DOCKER_USERNAME}/financial-python-api:${VERSION} \
    -t ${DOCKER_USERNAME}/financial-python-api:${GIT_COMMIT} \
    --build-arg BUILD_DATE="${BUILD_DATE}" \
    --build-arg GIT_COMMIT="${GIT_COMMIT}" \
    --platform linux/amd64 \
    -f LLM/Dockerfile \
    ./LLM

echo -e "${GREEN}✅ Python API image built${NC}"

# List built images
echo ""
echo "Built images:"
docker images | grep -E "financial-(java-service|python-api)"

# Push to Docker Hub
echo ""
read -p "Push to Docker Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Logging in to Docker Hub...${NC}"
    docker login
    
    echo -e "${YELLOW}Pushing Java service...${NC}"
    docker push ${DOCKER_USERNAME}/financial-java-service:${VERSION}
    docker push ${DOCKER_USERNAME}/financial-java-service:${GIT_COMMIT}
    
    echo -e "${YELLOW}Pushing Python API...${NC}"
    docker push ${DOCKER_USERNAME}/financial-python-api:${VERSION}
    docker push ${DOCKER_USERNAME}/financial-python-api:${GIT_COMMIT}
    
    echo -e "${GREEN}✅ Images pushed to Docker Hub${NC}"
fi

# Push to AWS ECR (optional)
echo ""
read -p "Push to AWS ECR? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
    AWS_REGION="${AWS_REGION:-us-east-1}"
    ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    echo -e "${YELLOW}Logging in to AWS ECR...${NC}"
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${ECR_REGISTRY}
    
    # Create repositories if they don't exist
    aws ecr describe-repositories --repository-names financial-java-service --region ${AWS_REGION} 2>/dev/null || \
        aws ecr create-repository --repository-name financial-java-service --region ${AWS_REGION}
    
    aws ecr describe-repositories --repository-names financial-python-api --region ${AWS_REGION} 2>/dev/null || \
        aws ecr create-repository --repository-name financial-python-api --region ${AWS_REGION}
    
    # Tag for ECR
    docker tag ${DOCKER_USERNAME}/financial-java-service:${VERSION} \
        ${ECR_REGISTRY}/financial-java-service:${VERSION}
    docker tag ${DOCKER_USERNAME}/financial-java-service:${VERSION} \
        ${ECR_REGISTRY}/financial-java-service:${GIT_COMMIT}
    
    docker tag ${DOCKER_USERNAME}/financial-python-api:${VERSION} \
        ${ECR_REGISTRY}/financial-python-api:${VERSION}
    docker tag ${DOCKER_USERNAME}/financial-python-api:${VERSION} \
        ${ECR_REGISTRY}/financial-python-api:${GIT_COMMIT}
    
    # Push to ECR
    echo -e "${YELLOW}Pushing to ECR...${NC}"
    docker push ${ECR_REGISTRY}/financial-java-service:${VERSION}
    docker push ${ECR_REGISTRY}/financial-java-service:${GIT_COMMIT}
    docker push ${ECR_REGISTRY}/financial-python-api:${VERSION}
    docker push ${ECR_REGISTRY}/financial-python-api:${GIT_COMMIT}
    
    echo -e "${GREEN}✅ Images pushed to AWS ECR${NC}"
    echo "ECR URLs:"
    echo "  ${ECR_REGISTRY}/financial-java-service:${VERSION}"
    echo "  ${ECR_REGISTRY}/financial-python-api:${VERSION}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Docker Build Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Docker Hub URLs:"
echo "  docker pull ${DOCKER_USERNAME}/financial-java-service:${VERSION}"
echo "  docker pull ${DOCKER_USERNAME}/financial-python-api:${VERSION}"
echo ""
