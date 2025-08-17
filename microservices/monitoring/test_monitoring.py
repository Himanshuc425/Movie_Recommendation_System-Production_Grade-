import unittest
import requests
import json
import os
import sys
import time
from unittest.mock import patch, MagicMock

# Add parent directory to path to import monitoring_client
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.monitoring_client import MonitoringClient

class TestMonitoringService(unittest.TestCase):
    """Test cases for the monitoring service and client"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Set environment variables for testing
        os.environ["MONITORING_SERVICE_URL"] = "http://localhost:8080"
        
        # Initialize client
        cls.client = MonitoringClient(service_name="test-service")
    
    @patch('requests.post')
    def test_record_model_metrics(self, mock_post):
        """Test recording model metrics"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "recorded", "id": 1}
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Call the client method
        response = self.client.record_model_metrics(
            model_type="test_model",
            accuracy=0.85,
            latency=0.05,
            throughput=100.0,
            drift=0.02
        )
        
        # Verify the response
        self.assertEqual(response["status"], "recorded")
        self.assertEqual(response["id"], 1)
        
        # Verify the request
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:8080/metrics/model")
        self.assertEqual(kwargs["json"]["model_type"], "test_model")
        self.assertEqual(kwargs["json"]["accuracy"], 0.85)
        self.assertEqual(kwargs["json"]["latency"], 0.05)
        self.assertEqual(kwargs["json"]["throughput"], 100.0)
        self.assertEqual(kwargs["json"]["drift"], 0.02)
    
    @patch('requests.post')
    def test_record_data_quality_metrics(self, mock_post):
        """Test recording data quality metrics"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "recorded", "id": 1}
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Call the client method
        response = self.client.record_data_quality_metrics(
            completeness=0.98,
            accuracy=0.95,
            consistency=0.97,
            timeliness=0.99
        )
        
        # Verify the response
        self.assertEqual(response["status"], "recorded")
        self.assertEqual(response["id"], 1)
        
        # Verify the request
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:8080/metrics/data-quality")
        self.assertEqual(kwargs["json"]["completeness"], 0.98)
        self.assertEqual(kwargs["json"]["accuracy"], 0.95)
        self.assertEqual(kwargs["json"]["consistency"], 0.97)
        self.assertEqual(kwargs["json"]["timeliness"], 0.99)
        self.assertAlmostEqual(kwargs["json"]["overall_score"], 0.9725)
    
    @patch('requests.post')
    def test_record_user_experience_metrics(self, mock_post):
        """Test recording user experience metrics"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "recorded", "id": 1}
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Call the client method
        response = self.client.record_user_experience_metrics(
            user_id="user123",
            satisfaction_score=0.9,
            click_through_rate=0.3,
            dwell_time=45.0,
            conversion_rate=0.15
        )
        
        # Verify the response
        self.assertEqual(response["status"], "recorded")
        self.assertEqual(response["id"], 1)
        
        # Verify the request
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://localhost:8080/metrics/user-experience")
        self.assertEqual(kwargs["json"]["user_id"], "user123")
        self.assertEqual(kwargs["json"]["satisfaction_score"], 0.9)
        self.assertEqual(kwargs["json"]["click_through_rate"], 0.3)
        self.assertEqual(kwargs["json"]["dwell_time"], 45.0)
        self.assertEqual(kwargs["json"]["conversion_rate"], 0.15)
    
    @patch('requests.get')
    def test_get_alerts(self, mock_get):
        """Test getting alerts"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "alerts": [
                {
                    "id": "123",
                    "service": "test-service",
                    "severity": "high",
                    "message": "Test alert",
                    "status": "active"
                }
            ],
            "count": 1
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Call the client method
        response = self.client.get_alerts(status="active")
        
        # Verify the response
        self.assertEqual(response["count"], 1)
        self.assertEqual(response["alerts"][0]["service"], "test-service")
        
        # Verify the request
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "http://localhost:8080/alerts?status=active")
    
    @patch('requests.get')
    def test_get_dashboard_summary(self, mock_get):
        """Test getting dashboard summary"""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "system": {
                "cpu_percent": 25.0,
                "memory_percent": 40.0,
                "disk_percent": 30.0
            },
            "services": {
                "data-pipeline": "up",
                "model-training": "up",
                "inference-service": "up"
            },
            "alerts": {
                "active": 0,
                "acknowledged": 1,
                "resolved": 2
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Call the client method
        response = self.client.get_dashboard_summary()
        
        # Verify the response
        self.assertEqual(response["system"]["cpu_percent"], 25.0)
        self.assertEqual(response["services"]["data-pipeline"], "up")
        self.assertEqual(response["alerts"]["active"], 0)
        
        # Verify the request
        args, kwargs = mock_get.call_args
        self.assertEqual(args[0], "http://localhost:8080/dashboard/summary")

    @patch('requests.post')
    def test_request_exception_handling(self, mock_post):
        """Test handling of request exceptions"""
        # Setup mock to raise an exception
        mock_post.side_effect = requests.RequestException("Connection error")
        
        # Call the client method
        response = self.client.record_model_metrics(
            model_type="test_model",
            accuracy=0.85,
            latency=0.05,
            throughput=100.0
        )
        
        # Verify the response indicates an error
        self.assertEqual(response["status"], "error")
        self.assertIn("Connection error", response["message"])

class TestMonitoringIntegration(unittest.TestCase):
    """Integration tests for the monitoring service"""
    
    def setUp(self):
        """Set up test environment"""
        # Skip tests if INTEGRATION_TESTS environment variable is not set
        if not os.environ.get("INTEGRATION_TESTS"):
            self.skipTest("Skipping integration tests")
        
        # Initialize client
        self.client = MonitoringClient(service_name="test-service")
        
        # Wait for monitoring service to be available
        self._wait_for_service()
    
    def _wait_for_service(self, max_retries=5, retry_delay=1):
        """Wait for the monitoring service to be available"""
        for i in range(max_retries):
            try:
                response = requests.get(f"{self.client.monitoring_url}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(retry_delay)
        
        self.skipTest("Monitoring service is not available")
    
    def test_record_and_retrieve_metrics(self):
        """Test recording and retrieving metrics"""
        # Record model metrics
        response = self.client.record_model_metrics(
            model_type="integration_test_model",
            accuracy=0.85,
            latency=0.05,
            throughput=100.0
        )
        self.assertEqual(response["status"], "recorded")
        
        # Get dashboard summary
        summary = self.client.get_dashboard_summary()
        self.assertIn("system", summary)
        self.assertIn("services", summary)

if __name__ == "__main__":
    unittest.main()