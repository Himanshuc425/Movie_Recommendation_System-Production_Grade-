import json
import logging
import os
import socket
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MonitoringClient:
    """Client library for sending metrics to the monitoring service"""
    
    def __init__(self, service_name: str, monitoring_url: Optional[str] = None):
        """Initialize the monitoring client
        
        Args:
            service_name: Name of the service using this client
            monitoring_url: URL of the monitoring service, defaults to environment variable
        """
        self.service_name = service_name
        self.monitoring_url = monitoring_url or os.getenv("MONITORING_SERVICE_URL", "http://monitoring-service:8080")
        self.hostname = socket.gethostname()
        logger.info(f"Initialized monitoring client for {service_name} at {self.monitoring_url}")
    
    def _send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the monitoring service
        
        Args:
            endpoint: API endpoint to call
            data: Data to send
            
        Returns:
            Response from the monitoring service
        """
        url = f"{self.monitoring_url}{endpoint}"
        try:
            response = requests.post(url, json=data, timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to send metrics to {url}: {e}")
            return {"status": "error", "message": str(e)}
    
    def record_model_metrics(self, model_type: str, accuracy: float, latency: float, 
                           throughput: float, drift: Optional[float] = None) -> Dict[str, Any]:
        """Record model performance metrics
        
        Args:
            model_type: Type of model (e.g., 'ncf', 'gnn', 'multi_modal')
            accuracy: Model accuracy (0-1)
            latency: Prediction latency in seconds
            throughput: Predictions per second
            drift: Optional model drift metric
            
        Returns:
            Response from the monitoring service
        """
        data = {
            "model_type": model_type,
            "accuracy": accuracy,
            "latency": latency,
            "throughput": throughput,
            "timestamp": datetime.now().isoformat()
        }
        
        if drift is not None:
            data["drift"] = drift
        
        return self._send_request("/metrics/model", data)
    
    def record_data_quality_metrics(self, completeness: float, accuracy: float, 
                                  consistency: float, timeliness: float) -> Dict[str, Any]:
        """Record data quality metrics
        
        Args:
            completeness: Data completeness score (0-1)
            accuracy: Data accuracy score (0-1)
            consistency: Data consistency score (0-1)
            timeliness: Data timeliness score (0-1)
            
        Returns:
            Response from the monitoring service
        """
        # Calculate overall score as average of individual metrics
        overall_score = (completeness + accuracy + consistency + timeliness) / 4.0
        
        data = {
            "completeness": completeness,
            "accuracy": accuracy,
            "consistency": consistency,
            "timeliness": timeliness,
            "overall_score": overall_score,
            "timestamp": datetime.now().isoformat()
        }
        
        return self._send_request("/metrics/data-quality", data)
    
    def record_user_experience_metrics(self, user_id: str, satisfaction_score: float,
                                     click_through_rate: float, dwell_time: float,
                                     conversion_rate: Optional[float] = None) -> Dict[str, Any]:
        """Record user experience metrics
        
        Args:
            user_id: Unique identifier for the user
            satisfaction_score: User satisfaction score (0-1)
            click_through_rate: Click-through rate (0-1)
            dwell_time: Average time spent on recommendations (seconds)
            conversion_rate: Optional conversion rate (0-1)
            
        Returns:
            Response from the monitoring service
        """
        data = {
            "user_id": user_id,
            "satisfaction_score": satisfaction_score,
            "click_through_rate": click_through_rate,
            "dwell_time": dwell_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if conversion_rate is not None:
            data["conversion_rate"] = conversion_rate
        
        return self._send_request("/metrics/user-experience", data)
    
    def get_alerts(self, status: Optional[str] = None) -> Dict[str, Any]:
        """Get current alerts from the monitoring service
        
        Args:
            status: Optional filter for alert status ('active', 'acknowledged', 'resolved')
            
        Returns:
            Dictionary containing alerts
        """
        url = f"{self.monitoring_url}/alerts"
        if status:
            url += f"?status={status}"
        
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to get alerts from {url}: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get a summary of system status for dashboard
        
        Returns:
            Dictionary containing system status summary
        """
        url = f"{self.monitoring_url}/dashboard/summary"
        
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Failed to get dashboard summary from {url}: {e}")
            return {"status": "error", "message": str(e)}

# Example usage
def example_usage():
    # Initialize client
    client = MonitoringClient(service_name="inference-service")
    
    # Record model metrics
    client.record_model_metrics(
        model_type="ncf",
        accuracy=0.85,
        latency=0.05,
        throughput=100.0,
        drift=0.02
    )
    
    # Record data quality metrics
    client.record_data_quality_metrics(
        completeness=0.98,
        accuracy=0.95,
        consistency=0.97,
        timeliness=0.99
    )
    
    # Record user experience metrics
    client.record_user_experience_metrics(
        user_id="user123",
        satisfaction_score=0.9,
        click_through_rate=0.3,
        dwell_time=45.0,
        conversion_rate=0.15
    )
    
    # Get active alerts
    alerts = client.get_alerts(status="active")
    print(f"Active alerts: {alerts}")
    
    # Get dashboard summary
    summary = client.get_dashboard_summary()
    print(f"Dashboard summary: {summary}")

if __name__ == "__main__":
    example_usage()