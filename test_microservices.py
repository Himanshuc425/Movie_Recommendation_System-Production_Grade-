import requests
import json
import time
import os
import sys

def test_data_pipeline():
    """Test the data pipeline service"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        if response.status_code == 200:
            print("✅ Data Pipeline Service is running")
            return True
        else:
            print("❌ Data Pipeline Service returned status code:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Data Pipeline Service is not accessible:", str(e))
        return False

def test_model_training():
    """Test the model training service"""
    try:
        response = requests.get("http://localhost:8081/health", timeout=2)
        if response.status_code == 200:
            print("✅ Model Training Service is running")
            return True
        else:
            print("❌ Model Training Service returned status code:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Model Training Service is not accessible:", str(e))
        return False

def test_inference_service():
    """Test the inference service"""
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8082/health", timeout=2)
        if response.status_code != 200:
            print("❌ Inference Service health check failed with status code:", response.status_code)
            return False
        
        # Test recommendation endpoint
        test_data = {
            "user_id": 42,
            "n": 5,
            "model": "ncf"
        }
        response = requests.post(
            "http://localhost:8082/recommendations", 
            json=test_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Inference Service is running and returning recommendations")
            return True
        else:
            print("❌ Inference Service recommendation endpoint returned status code:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Inference Service is not accessible:", str(e))
        return False

def test_mlflow():
    """Test MLflow tracking server"""
    try:
        response = requests.get("http://localhost:5000/api/2.0/mlflow/experiments/list", timeout=2)
        if response.status_code == 200:
            print("✅ MLflow tracking server is running")
            return True
        else:
            print("❌ MLflow tracking server returned status code:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ MLflow tracking server is not accessible:", str(e))
        return False

def test_prometheus():
    """Test Prometheus monitoring"""
    try:
        response = requests.get("http://localhost:9090/-/healthy", timeout=2)
        if response.status_code == 200:
            print("✅ Prometheus is running")
            return True
        else:
            print("❌ Prometheus returned status code:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Prometheus is not accessible:", str(e))
        return False

def main():
    print("\n🔍 Testing Microservices Architecture\n")
    
    # Test each component
    data_pipeline_ok = test_data_pipeline()
    model_training_ok = test_model_training()
    inference_service_ok = test_inference_service()
    mlflow_ok = test_mlflow()
    prometheus_ok = test_prometheus()
    
    # Summary
    print("\n📊 Test Summary:")
    total = 5
    passed = sum([data_pipeline_ok, model_training_ok, inference_service_ok, mlflow_ok, prometheus_ok])
    
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("\n✅ All microservices are running correctly!")
        return 0
    else:
        print("\n⚠️ Some microservices are not running or not accessible.")
        print("Please make sure all services are started with 'docker-compose up' or deployed to Kubernetes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())