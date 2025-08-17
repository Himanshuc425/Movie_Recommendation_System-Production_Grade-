#!/bin/bash

# Script to deploy the recommendation system microservices architecture

set -e

echo "🚀 Deploying Recommendation System Microservices"

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check for required tools
if ! command_exists docker; then
  echo "❌ Docker is not installed. Please install Docker first."
  exit 1
fi

# Determine deployment mode
if [ "$1" = "kubernetes" ]; then
  # Check for kubectl
  if ! command_exists kubectl; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
  fi
  
  echo "\n📦 Deploying to Kubernetes..."
  
  # Create namespaces
  echo "\n🔧 Creating recommendation-system namespace..."
  kubectl create namespace recommendation-system --dry-run=client -o yaml | kubectl apply -f -
  
  # Apply Kubernetes configurations
  echo "\n🔧 Applying Kubernetes configurations..."
  kubectl apply -f kubernetes/kafka-zookeeper-deployment.yaml -n recommendation-system
  echo "✅ Kafka and Zookeeper deployed"
  
  kubectl apply -f kubernetes/mlflow-redis-deployment.yaml -n recommendation-system
  echo "✅ MLflow and Redis deployed"
  
  kubectl apply -f kubernetes/prometheus-grafana-deployment.yaml -n recommendation-system
  echo "✅ Prometheus and Grafana deployed"
  
  kubectl apply -f kubernetes/data-pipeline-deployment.yaml -n recommendation-system
  echo "✅ Data Pipeline service deployed"
  
  kubectl apply -f kubernetes/model-training-deployment.yaml -n recommendation-system
  echo "✅ Model Training service deployed"
  
  kubectl apply -f kubernetes/inference-service-deployment.yaml -n recommendation-system
  echo "✅ Inference service deployed"
  
  kubectl apply -f kubernetes/monitoring-service-deployment.yaml -n recommendation-system
  echo "✅ Monitoring service deployed"
  
  kubectl apply -f kubernetes/ingress.yaml -n recommendation-system
  echo "✅ Ingress configured"
  
  # Wait for deployments to be ready
  echo "\n⏳ Waiting for deployments to be ready..."
  kubectl wait --for=condition=available --timeout=300s deployment/data-pipeline -n recommendation-system
  kubectl wait --for=condition=available --timeout=300s deployment/model-training -n recommendation-system
  kubectl wait --for=condition=available --timeout=300s deployment/inference-service -n recommendation-system
  kubectl wait --for=condition=available --timeout=300s deployment/monitoring-service -n recommendation-system
  
  echo "\n✅ All services deployed successfully to Kubernetes!"
  echo "\n📊 Access the services:"
  echo "- Recommendations API: http://recommender.example.com/api/recommendations"
  echo "- MLflow: http://recommender.example.com/mlflow"
  echo "- Grafana: http://recommender.example.com/grafana"
  echo "- Prometheus: http://recommender.example.com/prometheus"
  echo "- Monitoring Service: http://recommender.example.com/monitoring"
  
else
  # Docker Compose deployment
  if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
  fi
  
  echo "\n📦 Deploying with Docker Compose..."
  
  # Build and start the services
  docker-compose build
  docker-compose up -d
  
  echo "\n✅ All services deployed successfully with Docker Compose!"
  echo "\n📊 Access the services:"
  echo "- Data Pipeline API: http://localhost:8081"
  echo "- Model Training API: http://localhost:8082"
  echo "- Recommendations API: http://localhost:8080"
  echo "- Monitoring Service: http://localhost:8083"
  echo "- MLflow: http://localhost:5000"
  echo "- Grafana: http://localhost:3000"
  echo "- Prometheus: http://localhost:9090"
  
  # Run tests
  echo "\n🧪 Running tests to verify deployment..."
  python test_microservices.py
fi

echo "\n🎉 Deployment completed!"