# PowerShell script to deploy the recommendation system microservices architecture

$ErrorActionPreference = "Stop"

Write-Host "🚀 Deploying Recommendation System Microservices" -ForegroundColor Cyan

# Function to check if a command exists
function Test-CommandExists {
    param ($command)
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# Check for required tools
if (-not (Test-CommandExists docker)) {
    Write-Host "❌ Docker is not installed. Please install Docker first." -ForegroundColor Red
    exit 1
}

# Determine deployment mode
if ($args[0] -eq "kubernetes") {
    # Check for kubectl
    if (-not (Test-CommandExists kubectl)) {
        Write-Host "❌ kubectl is not installed. Please install kubectl first." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`n📦 Deploying to Kubernetes..." -ForegroundColor Cyan
    
    # Create namespaces
    Write-Host "`n🔧 Creating recommendation-system namespace..." -ForegroundColor Yellow
    kubectl create namespace recommendation-system --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes configurations
    Write-Host "`n🔧 Applying Kubernetes configurations..." -ForegroundColor Yellow
    kubectl apply -f kubernetes/kafka-zookeeper-deployment.yaml -n recommendation-system
    Write-Host "✅ Kafka and Zookeeper deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/mlflow-redis-deployment.yaml -n recommendation-system
    Write-Host "✅ MLflow and Redis deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/prometheus-grafana-deployment.yaml -n recommendation-system
    Write-Host "✅ Prometheus and Grafana deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/data-pipeline-deployment.yaml -n recommendation-system
    Write-Host "✅ Data Pipeline service deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/model-training-deployment.yaml -n recommendation-system
    Write-Host "✅ Model Training service deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/inference-service-deployment.yaml -n recommendation-system
    Write-Host "✅ Inference service deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/monitoring-service-deployment.yaml -n recommendation-system
    Write-Host "✅ Monitoring service deployed" -ForegroundColor Green
    
    kubectl apply -f kubernetes/ingress.yaml -n recommendation-system
    Write-Host "✅ Ingress configured" -ForegroundColor Green
    
    # Wait for deployments to be ready
    Write-Host "`n⏳ Waiting for deployments to be ready..." -ForegroundColor Yellow
    kubectl wait --for=condition=available --timeout=300s deployment/data-pipeline -n recommendation-system
    kubectl wait --for=condition=available --timeout=300s deployment/model-training -n recommendation-system
    kubectl wait --for=condition=available --timeout=300s deployment/inference-service -n recommendation-system
    kubectl wait --for=condition=available --timeout=300s deployment/monitoring-service -n recommendation-system
    
    Write-Host "`n✅ All services deployed successfully to Kubernetes!" -ForegroundColor Green
    Write-Host "`n📊 Access the services:" -ForegroundColor Cyan
    Write-Host "- Recommendations API: http://recommender.example.com/api/recommendations"
    Write-Host "- MLflow: http://recommender.example.com/mlflow"
    Write-Host "- Grafana: http://recommender.example.com/grafana"
    Write-Host "- Prometheus: http://recommender.example.com/prometheus"
    Write-Host "- Monitoring Service: http://recommender.example.com/monitoring"
    
} else {
    # Docker Compose deployment
    if (-not (Test-CommandExists docker-compose)) {
        Write-Host "❌ Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
        exit 1
    }
    
    Write-Host "`n📦 Deploying with Docker Compose..." -ForegroundColor Cyan
    
    # Build and start the services
    docker-compose build
    docker-compose up -d
    
    Write-Host "`n✅ All services deployed successfully with Docker Compose!" -ForegroundColor Green
    Write-Host "`n📊 Access the services:" -ForegroundColor Cyan
    Write-Host "- Data Pipeline API: http://localhost:8081"
    Write-Host "- Model Training API: http://localhost:8082"
    Write-Host "- Recommendations API: http://localhost:8080"
    Write-Host "- Monitoring Service: http://localhost:8083"
    Write-Host "- MLflow: http://localhost:5000"
    Write-Host "- Grafana: http://localhost:3000"
    Write-Host "- Prometheus: http://localhost:9090"
    
    # Run tests
    Write-Host "`n🧪 Running tests to verify deployment..." -ForegroundColor Yellow
    python test_microservices.py
}

Write-Host "`n🎉 Deployment completed!" -ForegroundColor Green