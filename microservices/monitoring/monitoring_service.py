import time
import json
import logging
import os
import socket
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from prometheus_client import CollectorRegistry, push_to_gateway
import psutil
import requests
from starlette.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SERVICE_NAME = "monitoring-service"
SERVICE_VERSION = "1.0.0"
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8000"))
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://prometheus-pushgateway:9091")
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "15"))  # seconds

# Microservices to monitor
MICROSERVICES = {
    "data-pipeline": os.getenv("DATA_PIPELINE_URL", "http://data-pipeline:8080"),
    "model-training": os.getenv("MODEL_TRAINING_URL", "http://model-training:8080"),
    "inference-service": os.getenv("INFERENCE_SERVICE_URL", "http://inference-service:8080")
}

# Create FastAPI app
app = FastAPI(
    title="Recommendation System Monitoring Service",
    description="Comprehensive monitoring and observability for the recommendation system",
    version=SERVICE_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Prometheus registry
registry = CollectorRegistry()

# Define Prometheus metrics
# System metrics
system_cpu_usage = Gauge('system_cpu_usage', 'System CPU usage percentage', registry=registry)
system_memory_usage = Gauge('system_memory_usage', 'System memory usage percentage', registry=registry)
system_disk_usage = Gauge('system_disk_usage', 'System disk usage percentage', registry=registry)

# Service health metrics
service_health = Gauge('service_health', 'Service health status (1=up, 0=down)', 
                      ['service'], registry=registry)
service_response_time = Histogram('service_response_time', 'Service response time in seconds',
                                 ['service'], registry=registry)

# Model metrics
model_prediction_count = Counter('model_prediction_count', 'Number of model predictions',
                               ['model_type'], registry=registry)
model_prediction_latency = Histogram('model_prediction_latency', 
                                   'Model prediction latency in seconds',
                                   ['model_type'], registry=registry)
model_accuracy = Gauge('model_accuracy', 'Model accuracy metric', 
                      ['model_type'], registry=registry)
model_drift = Gauge('model_drift', 'Model drift metric', 
                   ['model_type'], registry=registry)

# Data pipeline metrics
data_pipeline_records_processed = Counter('data_pipeline_records_processed', 
                                        'Number of records processed by data pipeline',
                                        registry=registry)
data_pipeline_processing_time = Histogram('data_pipeline_processing_time',
                                        'Data pipeline processing time in seconds',
                                        registry=registry)
data_quality_score = Gauge('data_quality_score', 'Data quality score', registry=registry)

# User experience metrics
user_recommendation_satisfaction = Gauge('user_recommendation_satisfaction',
                                       'User satisfaction with recommendations (0-1)',
                                       registry=registry)
recommendation_diversity = Gauge('recommendation_diversity',
                               'Diversity of recommendations (0-1)',
                               registry=registry)
recommendation_coverage = Gauge('recommendation_coverage',
                              'Coverage of recommendations (0-1)',
                              registry=registry)

# Alert metrics
alert_count = Counter('alert_count', 'Number of alerts generated',
                     ['severity', 'type'], registry=registry)

# Pydantic models for API
class HealthStatus(BaseModel):
    service: str
    status: str
    version: str
    timestamp: str

class MetricValue(BaseModel):
    name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: str

class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service: str
    severity: str
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, acknowledged, resolved

class ModelMetrics(BaseModel):
    model_type: str
    accuracy: float
    latency: float
    throughput: float
    drift: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class DataQualityMetrics(BaseModel):
    completeness: float
    accuracy: float
    consistency: float
    timeliness: float
    overall_score: float
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class UserExperienceMetrics(BaseModel):
    user_id: str
    satisfaction_score: float
    click_through_rate: float
    dwell_time: float
    conversion_rate: Optional[float] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

# In-memory storage for alerts and metrics history
alerts_db: List[Alert] = []
metrics_history: Dict[str, List[Dict[str, Any]]] = {
    "system": [],
    "service_health": [],
    "model": [],
    "data_quality": [],
    "user_experience": []
}

# Background monitoring task
def monitor_system_metrics():
    """Collect and store system metrics"""
    while True:
        try:
            # Collect CPU, memory, and disk usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Update Prometheus metrics
            system_cpu_usage.set(cpu_percent)
            system_memory_usage.set(memory_percent)
            system_disk_usage.set(disk_percent)
            
            # Store in history
            timestamp = datetime.now().isoformat()
            metrics_history["system"].append({
                "timestamp": timestamp,
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent
            })
            
            # Trim history if it gets too large
            if len(metrics_history["system"]) > 1000:
                metrics_history["system"] = metrics_history["system"][-1000:]
            
            # Check for alerts
            if cpu_percent > 90:
                generate_alert("system", "high", f"High CPU usage: {cpu_percent}%", 
                              "system_cpu_usage", cpu_percent, 90)
            
            if memory_percent > 90:
                generate_alert("system", "high", f"High memory usage: {memory_percent}%", 
                              "system_memory_usage", memory_percent, 90)
            
            if disk_percent > 90:
                generate_alert("system", "high", f"High disk usage: {disk_percent}%", 
                              "system_disk_usage", disk_percent, 90)
            
            # Push to Prometheus Pushgateway if configured
            if PUSHGATEWAY_URL:
                try:
                    push_to_gateway(PUSHGATEWAY_URL, job='system_metrics', registry=registry)
                except Exception as e:
                    logger.error(f"Failed to push to Pushgateway: {e}")
            
            # Wait for next monitoring interval
            time.sleep(MONITORING_INTERVAL)
        except Exception as e:
            logger.error(f"Error in system monitoring: {e}")
            time.sleep(MONITORING_INTERVAL)

def monitor_service_health():
    """Check health of all microservices"""
    while True:
        try:
            timestamp = datetime.now().isoformat()
            service_statuses = []
            
            for service_name, service_url in MICROSERVICES.items():
                try:
                    start_time = time.time()
                    response = requests.get(f"{service_url}/health", timeout=5)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        status = "up"
                        service_health.labels(service=service_name).set(1)
                    else:
                        status = "down"
                        service_health.labels(service=service_name).set(0)
                        generate_alert(service_name, "high", 
                                      f"Service {service_name} returned status code {response.status_code}",
                                      "service_health", 0, 1)
                    
                    # Record response time
                    service_response_time.labels(service=service_name).observe(response_time)
                    
                    # Store service status
                    service_statuses.append({
                        "service": service_name,
                        "status": status,
                        "response_time": response_time,
                        "timestamp": timestamp
                    })
                    
                except requests.RequestException as e:
                    logger.error(f"Error checking {service_name} health: {e}")
                    service_health.labels(service=service_name).set(0)
                    service_statuses.append({
                        "service": service_name,
                        "status": "down",
                        "error": str(e),
                        "timestamp": timestamp
                    })
                    generate_alert(service_name, "high", 
                                  f"Service {service_name} is unreachable: {e}",
                                  "service_health", 0, 1)
            
            # Store in history
            metrics_history["service_health"].append({
                "timestamp": timestamp,
                "services": service_statuses
            })
            
            # Trim history if it gets too large
            if len(metrics_history["service_health"]) > 1000:
                metrics_history["service_health"] = metrics_history["service_health"][-1000:]
            
            # Push to Prometheus Pushgateway if configured
            if PUSHGATEWAY_URL:
                try:
                    push_to_gateway(PUSHGATEWAY_URL, job='service_health', registry=registry)
                except Exception as e:
                    logger.error(f"Failed to push to Pushgateway: {e}")
            
            # Wait for next monitoring interval
            time.sleep(MONITORING_INTERVAL)
        except Exception as e:
            logger.error(f"Error in service health monitoring: {e}")
            time.sleep(MONITORING_INTERVAL)

def generate_alert(service: str, severity: str, message: str, metric: str, value: float, threshold: float):
    """Generate and store an alert"""
    alert = Alert(
        service=service,
        severity=severity,
        message=message,
        metric=metric,
        value=value,
        threshold=threshold
    )
    
    # Check if a similar alert already exists and is active
    for existing_alert in alerts_db:
        if (existing_alert.service == service and 
            existing_alert.metric == metric and 
            existing_alert.status == "active"):
            # Update the existing alert instead of creating a new one
            existing_alert.value = value
            existing_alert.timestamp = datetime.now().isoformat()
            logger.info(f"Updated alert: {existing_alert}")
            
            # Increment alert counter
            alert_count.labels(severity=severity, type=metric).inc()
            return existing_alert
    
    # If no similar alert exists, create a new one
    alerts_db.append(alert)
    logger.warning(f"New alert generated: {alert}")
    
    # Increment alert counter
    alert_count.labels(severity=severity, type=metric).inc()
    
    # TODO: Send notification (email, Slack, etc.)
    
    return alert

# API endpoints
@app.get("/")
async def root():
    return {"message": f"{SERVICE_NAME} is running", "version": SERVICE_VERSION}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics():
    """Get current metrics values"""
    return {
        "system": {
            "cpu_usage": system_cpu_usage._value.get(),
            "memory_usage": system_memory_usage._value.get(),
            "disk_usage": system_disk_usage._value.get(),
        },
        "services": {
            service: {
                "health": service_health.labels(service=service)._value.get(),
                # Note: Histograms don't have a simple current value to report
            } for service in MICROSERVICES.keys()
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics/history/{metric_type}")
async def get_metrics_history(metric_type: str):
    """Get historical metrics data"""
    if metric_type not in metrics_history:
        raise HTTPException(status_code=404, detail=f"Metric type '{metric_type}' not found")
    
    return {"history": metrics_history[metric_type]}

@app.post("/metrics/model")
async def record_model_metrics(metrics: ModelMetrics):
    """Record model performance metrics"""
    # Update Prometheus metrics
    model_prediction_count.labels(model_type=metrics.model_type).inc()
    model_prediction_latency.labels(model_type=metrics.model_type).observe(metrics.latency)
    model_accuracy.labels(model_type=metrics.model_type).set(metrics.accuracy)
    
    if metrics.drift is not None:
        model_drift.labels(model_type=metrics.model_type).set(metrics.drift)
    
    # Store in history
    metrics_history["model"].append(metrics.dict())
    
    # Trim history if it gets too large
    if len(metrics_history["model"]) > 1000:
        metrics_history["model"] = metrics_history["model"][-1000:]
    
    # Check for alerts
    if metrics.accuracy < 0.7:
        generate_alert("model-training", "medium", 
                      f"Low model accuracy for {metrics.model_type}: {metrics.accuracy}",
                      "model_accuracy", metrics.accuracy, 0.7)
    
    if metrics.drift is not None and metrics.drift > 0.2:
        generate_alert("model-training", "high", 
                      f"High model drift detected for {metrics.model_type}: {metrics.drift}",
                      "model_drift", metrics.drift, 0.2)
    
    return {"status": "recorded", "id": len(metrics_history["model"])}

@app.post("/metrics/data-quality")
async def record_data_quality_metrics(metrics: DataQualityMetrics):
    """Record data quality metrics"""
    # Update Prometheus metrics
    data_quality_score.set(metrics.overall_score)
    
    # Store in history
    metrics_history["data_quality"].append(metrics.dict())
    
    # Trim history if it gets too large
    if len(metrics_history["data_quality"]) > 1000:
        metrics_history["data_quality"] = metrics_history["data_quality"][-1000:]
    
    # Check for alerts
    if metrics.overall_score < 0.8:
        generate_alert("data-pipeline", "medium", 
                      f"Low data quality score: {metrics.overall_score}",
                      "data_quality_score", metrics.overall_score, 0.8)
    
    return {"status": "recorded", "id": len(metrics_history["data_quality"])}

@app.post("/metrics/user-experience")
async def record_user_experience_metrics(metrics: UserExperienceMetrics):
    """Record user experience metrics"""
    # Update Prometheus metrics
    user_recommendation_satisfaction.set(metrics.satisfaction_score)
    
    # Store in history
    metrics_history["user_experience"].append(metrics.dict())
    
    # Trim history if it gets too large
    if len(metrics_history["user_experience"]) > 1000:
        metrics_history["user_experience"] = metrics_history["user_experience"][-1000:]
    
    # Check for alerts
    if metrics.satisfaction_score < 0.6:
        generate_alert("inference-service", "low", 
                      f"Low user satisfaction score: {metrics.satisfaction_score}",
                      "user_recommendation_satisfaction", metrics.satisfaction_score, 0.6)
    
    return {"status": "recorded", "id": len(metrics_history["user_experience"])}

@app.get("/alerts")
async def get_alerts(status: Optional[str] = None):
    """Get all alerts, optionally filtered by status"""
    if status:
        filtered_alerts = [alert for alert in alerts_db if alert.status == status]
        return {"alerts": filtered_alerts, "count": len(filtered_alerts)}
    else:
        return {"alerts": alerts_db, "count": len(alerts_db)}

@app.put("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    for alert in alerts_db:
        if alert.id == alert_id and alert.status == "active":
            alert.status = "acknowledged"
            return {"status": "success", "alert": alert}
    
    raise HTTPException(status_code=404, detail=f"Active alert with ID {alert_id} not found")

@app.put("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Resolve an alert"""
    for alert in alerts_db:
        if alert.id == alert_id and alert.status in ["active", "acknowledged"]:
            alert.status = "resolved"
            return {"status": "success", "alert": alert}
    
    raise HTTPException(status_code=404, detail=f"Active or acknowledged alert with ID {alert_id} not found")

@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get a summary of system status for dashboard"""
    # Get latest system metrics
    system_metrics = metrics_history["system"][-1] if metrics_history["system"] else {}
    
    # Get service health status
    service_statuses = {}
    if metrics_history["service_health"]:
        latest_health = metrics_history["service_health"][-1]
        for service in latest_health.get("services", []):
            service_statuses[service["service"]] = service["status"]
    
    # Count alerts by status
    alert_counts = {
        "active": len([a for a in alerts_db if a.status == "active"]),
        "acknowledged": len([a for a in alerts_db if a.status == "acknowledged"]),
        "resolved": len([a for a in alerts_db if a.status == "resolved"])
    }
    
    # Get latest model metrics
    model_metrics = {}
    for metric in metrics_history["model"][-10:] if metrics_history["model"] else []:
        model_type = metric["model_type"]
        if model_type not in model_metrics:
            model_metrics[model_type] = []
        model_metrics[model_type].append(metric)
    
    return {
        "system": system_metrics,
        "services": service_statuses,
        "alerts": alert_counts,
        "models": model_metrics,
        "timestamp": datetime.now().isoformat()
    }

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track request processing time"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Start background monitoring tasks
@app.on_event("startup")
def startup_event():
    # Start Prometheus HTTP server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    
    # Start background monitoring threads
    threading.Thread(target=monitor_system_metrics, daemon=True).start()
    threading.Thread(target=monitor_service_health, daemon=True).start()
    logger.info("Background monitoring tasks started")

# Main entry point
if __name__ == "__main__":
    uvicorn.run("monitoring_service:app", host="0.0.0.0", port=8080, reload=True)