#!/bin/bash

# Complete Kubernetes Deployment Script
# Financial Intelligence Platform

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    print_header "Checking Prerequisites"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl first."
        exit 1
    fi
    print_success "kubectl found: $(kubectl version --client --short 2>/dev/null || echo 'installed')"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    print_success "Connected to Kubernetes cluster"
    
    # Check Docker images (optional warning)
    echo ""
    print_warning "Make sure Docker images are pushed to registry:"
    print_warning "  - hermantqin/financial-java-service:latest"
    print_warning "  - hermantqin/financial-python-api:latest"
    echo ""
}

deploy_namespace() {
    print_header "Step 1/7: Creating Namespace"
    kubectl apply -f k8s/namespace.yaml
    print_success "Namespace created"
}

deploy_configmap() {
    print_header "Step 2/7: Applying ConfigMap"
    kubectl apply -f k8s/configmap.yaml
    print_success "ConfigMap applied"
}

deploy_secrets() {
    print_header "Step 3/7: Creating Secrets"
    
    if kubectl get secret financial-platform-secrets -n financial-platform &> /dev/null; then
        print_warning "Secrets already exist. Skipping creation."
        echo "To update secrets, delete first: kubectl delete secret financial-platform-secrets -n financial-platform"
    else
        print_warning "Using template secrets from k8s/secrets.yaml"
        print_warning "⚠️  IMPORTANT: Replace with actual secrets in production!"
        kubectl apply -f k8s/secrets.yaml
        print_success "Secrets created"
    fi
}

deploy_postgres() {
    print_header "Step 4/7: Deploying PostgreSQL Database"
    
    kubectl apply -f k8s/postgres-deployment.yaml
    
    echo "Waiting for PostgreSQL to be ready (timeout: 5 minutes)..."
    if kubectl wait --for=condition=ready pod -l app=postgres -n financial-platform --timeout=300s; then
        print_success "PostgreSQL is ready"
    else
        print_error "PostgreSQL deployment timeout"
        echo "Check logs: kubectl logs -l app=postgres -n financial-platform"
        exit 1
    fi
}

deploy_java_service() {
    print_header "Step 5/7: Deploying Java Transaction Service"
    
    kubectl apply -f k8s/java-service-deployment.yaml
    
    echo "Waiting for Java service rollout (timeout: 5 minutes)..."
    if kubectl rollout status deployment/java-service -n financial-platform --timeout=300s; then
        print_success "Java service deployed"
    else
        print_error "Java service deployment timeout"
        echo "Check logs: kubectl logs -l app=java-service -n financial-platform"
        exit 1
    fi
}

deploy_python_api() {
    print_header "Step 6/7: Deploying Python ML/LLM API"
    
    kubectl apply -f k8s/python-api-deployment.yaml
    
    echo "Waiting for Python API rollout (timeout: 5 minutes)..."
    if kubectl rollout status deployment/python-api -n financial-platform --timeout=300s; then
        print_success "Python API deployed"
    else
        print_error "Python API deployment timeout"
        echo "Check logs: kubectl logs -l app=python-api -n financial-platform"
        exit 1
    fi
}

deploy_ingress() {
    print_header "Step 7/7: Configuring Ingress"
    
    kubectl apply -f k8s/ingress.yaml
    print_success "Ingress configured"
}

verify_deployment() {
    print_header "Verifying Deployment"
    
    echo ""
    echo "📊 All Resources:"
    kubectl get all -n financial-platform
    
    echo ""
    echo "🔍 Ingress Status:"
    kubectl get ingress -n financial-platform
    
    echo ""
    echo "🏥 Pod Health:"
    kubectl get pods -n financial-platform -o wide
    
    echo ""
    echo "📈 HPA Status:"
    kubectl get hpa -n financial-platform
}

run_health_checks() {
    print_header "Running Health Checks"
    
    echo "Checking Java service health..."
    if kubectl run test-java --rm -i --restart=Never --image=curlimages/curl -n financial-platform -- \
        curl -f http://java-service:8080/health --max-time 10; then
        print_success "Java service is healthy"
    else
        print_warning "Java service health check failed (this is normal if service is still starting)"
    fi
    
    echo ""
    echo "Checking Python API health..."
    if kubectl run test-python --rm -i --restart=Never --image=curlimages/curl -n financial-platform -- \
        curl -f http://python-api:8000/health --max-time 10; then
        print_success "Python API is healthy"
    else
        print_warning "Python API health check failed (this is normal if service is still starting)"
    fi
}

print_useful_commands() {
    print_header "Useful Commands"
    
    echo ""
    echo "📋 View Resources:"
    echo "  kubectl get all -n financial-platform"
    echo ""
    echo "📝 View Logs:"
    echo "  kubectl logs -f deployment/java-service -n financial-platform"
    echo "  kubectl logs -f deployment/python-api -n financial-platform"
    echo "  kubectl logs -f deployment/postgres -n financial-platform"
    echo ""
    echo "🔍 Describe Resources:"
    echo "  kubectl describe pod <pod-name> -n financial-platform"
    echo "  kubectl describe svc <service-name> -n financial-platform"
    echo ""
    echo "🚪 Shell Access:"
    echo "  kubectl exec -it deployment/java-service -n financial-platform -- /bin/sh"
    echo "  kubectl exec -it deployment/python-api -n financial-platform -- /bin/bash"
    echo ""
    echo "🔌 Port Forwarding (local testing):"
    echo "  kubectl port-forward svc/java-service 8080:8080 -n financial-platform"
    echo "  kubectl port-forward svc/python-api 8000:8000 -n financial-platform"
    echo ""
    echo "📊 Monitoring:"
    echo "  kubectl top pods -n financial-platform"
    echo "  kubectl top nodes"
    echo ""
    echo "🗑️  Cleanup:"
    echo "  kubectl delete namespace financial-platform"
    echo ""
}

# Main execution
main() {
    echo ""
    print_header "🚀 Financial Intelligence Platform - K8s Deployment"
    echo ""
    
    check_prerequisites
    deploy_namespace
    deploy_configmap
    deploy_secrets
    deploy_postgres
    deploy_java_service
    deploy_python_api
    deploy_ingress
    
    echo ""
    verify_deployment
    
    echo ""
    echo "⏳ Waiting 30 seconds for services to fully initialize..."
    sleep 30
    
    echo ""
    run_health_checks
    
    echo ""
    print_useful_commands
    
    echo ""
    print_success "🎉 Deployment Complete!"
    echo ""
}

# Run main function
main
