#!/bin/bash

# AI IDE Deployment Script
# Usage: ./deploy.sh <environment> <version>

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    print_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

print_header "Deploying AI IDE to $ENVIRONMENT environment (version: $VERSION)"

# Set environment-specific variables
case $ENVIRONMENT in
    staging)
        NAMESPACE="ai-ide-staging"
        DOMAIN="staging.ai-ide.dev"
        REPLICAS=1
        ;;
    production)
        NAMESPACE="ai-ide-production"
        DOMAIN="ai-ide.dev"
        REPLICAS=3
        ;;
esac

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed"
    exit 1
fi

if ! command -v helm &> /dev/null; then
    print_error "helm is not installed"
    exit 1
fi

# Create namespace if it doesn't exist
print_status "Creating namespace $NAMESPACE..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy PostgreSQL with pgvector
print_status "Deploying PostgreSQL database..."
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

helm upgrade --install postgres bitnami/postgresql \
    --namespace $NAMESPACE \
    --set auth.postgresPassword=aiide_${ENVIRONMENT} \
    --set auth.username=aiide \
    --set auth.password=aiide_${ENVIRONMENT} \
    --set auth.database=ai_ide \
    --set primary.persistence.size=20Gi \
    --set image.tag=15 \
    --set primary.initdb.scripts."init-pgvector\.sql"="CREATE EXTENSION IF NOT EXISTS vector;"

# Deploy Redis
print_status "Deploying Redis cache..."
helm upgrade --install redis bitnami/redis \
    --namespace $NAMESPACE \
    --set auth.enabled=false \
    --set master.persistence.size=5Gi

# Wait for database services
print_status "Waiting for database services to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n $NAMESPACE --timeout=300s

# Apply Kubernetes manifests
print_status "Applying Kubernetes manifests..."

# Create ConfigMap for application configuration
kubectl create configmap ai-ide-config \
    --namespace=$NAMESPACE \
    --from-file=$SCRIPT_DIR/config/$ENVIRONMENT/ \
    --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
kubectl create secret generic ai-ide-secrets \
    --namespace=$NAMESPACE \
    --from-literal=database-password=aiide_${ENVIRONMENT} \
    --from-literal=jwt-secret=$(openssl rand -base64 32) \
    --from-literal=openai-api-key=${OPENAI_API_KEY:-""} \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply deployment manifests
envsubst < $SCRIPT_DIR/k8s/backend-deployment.yaml | kubectl apply -f -
envsubst < $SCRIPT_DIR/k8s/websearch-deployment.yaml | kubectl apply -f -
envsubst < $SCRIPT_DIR/k8s/rag-deployment.yaml | kubectl apply -f -
envsubst < $SCRIPT_DIR/k8s/mcp-deployment.yaml | kubectl apply -f -
envsubst < $SCRIPT_DIR/k8s/monitoring-deployment.yaml | kubectl apply -f -

# Apply services
kubectl apply -f $SCRIPT_DIR/k8s/services.yaml -n $NAMESPACE

# Apply ingress
envsubst < $SCRIPT_DIR/k8s/ingress.yaml | kubectl apply -f -

# Wait for deployments to be ready
print_status "Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/ai-ide-backend -n $NAMESPACE --timeout=600s
kubectl wait --for=condition=available deployment/ai-ide-websearch -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=available deployment/ai-ide-rag -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=available deployment/ai-ide-mcp -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=available deployment/ai-ide-monitoring -n $NAMESPACE --timeout=300s

# Run database migrations
print_status "Running database migrations..."
kubectl run migration-job-$(date +%s) \
    --namespace=$NAMESPACE \
    --image=ghcr.io/$GITHUB_REPOSITORY/ai-ide:$VERSION-backend \
    --restart=Never \
    --env="DATABASE_URL=postgresql://aiide:aiide_${ENVIRONMENT}@postgres-postgresql:5432/ai_ide" \
    --command -- python database/migrations.py

# Verify deployment
print_status "Verifying deployment..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Run health checks
print_status "Running health checks..."
BACKEND_URL="https://$DOMAIN"

for i in {1..30}; do
    if curl -f -s "$BACKEND_URL/health" > /dev/null; then
        print_status "Health check passed!"
        break
    fi
    if [ $i -eq 30 ]; then
        print_error "Health check failed after 30 attempts"
        exit 1
    fi
    print_status "Waiting for service to be ready... (attempt $i/30)"
    sleep 10
done

# Update MCP server configurations
print_status "Updating MCP server configurations..."
kubectl exec -n $NAMESPACE deployment/ai-ide-mcp -- python mcp_server_framework.py --update-configs

print_header "✅ Deployment to $ENVIRONMENT completed successfully!"
print_status "Application URL: https://$DOMAIN"
print_status "Monitoring Dashboard: https://$DOMAIN/monitoring"
print_status "API Documentation: https://$DOMAIN/docs"

# Show deployment summary
echo ""
echo "=== Deployment Summary ==="
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"
echo "Domain: $DOMAIN"
echo "Replicas: $REPLICAS"
echo "=========================="