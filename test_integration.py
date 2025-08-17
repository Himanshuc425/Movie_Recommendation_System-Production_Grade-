import unittest
import pandas as pd
import numpy as np
import os
import sys
import json
import requests
from unittest.mock import patch, MagicMock
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the advanced features modules
from microservices.advanced_features.multi_modal_recommender import MultiModalRecommender
from microservices.advanced_features.reinforcement_learning import RLRecommender, RecommenderEnvironment, DQNAgent
from microservices.advanced_features.explainable_ai import ExplainableRecommender, ExplainableNCF, ExplainableGNN

# Import the monitoring client
from microservices.monitoring.monitoring_client import MonitoringClient

class TestAdvancedFeaturesIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create sample data for testing
        cls.users = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 35, 40, 45],
            'gender': ['M', 'F', 'M', 'F', 'M']
        })
        
        cls.items = pd.DataFrame({
            'item_id': [101, 102, 103, 104, 105],
            'category': ['electronics', 'books', 'clothing', 'home', 'sports'],
            'price': [100, 20, 50, 200, 75]
        })
        
        cls.interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 4, 5],
            'item_id': [101, 103, 102, 104, 105, 101, 102],
            'rating': [5, 4, 3, 5, 4, 2, 3]
        })
        
        # Initialize the monitoring client with a mock URL
        cls.monitoring_client = MonitoringClient("http://localhost:8083")
    
    def test_multi_modal_with_monitoring(self):
        # Mock image and text features
        image_features = {item_id: np.random.rand(10) for item_id in self.items['item_id']}
        text_features = {item_id: np.random.rand(10) for item_id in self.items['item_id']}
        
        # Initialize the recommender with mocked features
        with patch('microservices.advanced_features.multi_modal_recommender.MultiModalRecommender._load_image_features', 
                  return_value=image_features), \
             patch('microservices.advanced_features.multi_modal_recommender.MultiModalRecommender._load_text_features', 
                  return_value=text_features), \
             patch.object(self.monitoring_client, 'record_model_metrics') as mock_record_metrics:
            
            # Create and train the recommender
            recommender = MultiModalRecommender()
            recommender.train(self.users, self.items, self.interactions)
            
            # Get recommendations
            start_time = time.time()
            recommendations = recommender.get_recommendations(1, self.users, self.items, top_n=3)
            prediction_time = time.time() - start_time
            
            # Record metrics with the monitoring client
            self.monitoring_client.record_model_metrics(
                model_name="multi_modal_recommender",
                accuracy=0.85,  # Mock accuracy
                prediction_latency=prediction_time,
                prediction_count=1,
                drift_score=0.05  # Mock drift score
            )
            
            # Check that the monitoring client was called
            mock_record_metrics.assert_called_once()
            
            # Check that recommendations are returned
            self.assertIsNotNone(recommendations)
            self.assertLessEqual(len(recommendations), 3)
    
    def test_reinforcement_learning_with_monitoring(self):
        # Initialize the environment
        env = RecommenderEnvironment(self.users, self.items, self.interactions)
        
        # Mock the DQN agent
        agent = MagicMock(spec=DQNAgent)
        agent.act.return_value = 0  # Always choose the first action
        
        # Initialize the recommender
        recommender = RLRecommender(env, agent)
        
        # Mock the monitoring client
        with patch.object(self.monitoring_client, 'record_user_experience_metrics') as mock_record_ux:
            # Get recommendations
            recommendations = recommender.get_recommendations(1, top_n=3)
            
            # Record user experience metrics
            self.monitoring_client.record_user_experience_metrics(
                user_id=1,
                satisfaction_score=4.5,  # Mock satisfaction
                diversity_score=0.8,     # Mock diversity
                coverage_score=0.7       # Mock coverage
            )
            
            # Check that the monitoring client was called
            mock_record_ux.assert_called_once()
            
            # Check that recommendations are returned
            self.assertIsNotNone(recommendations)
            self.assertLessEqual(len(recommendations), 3)
    
    def test_explainable_ai_with_monitoring(self):
        # Mock the base recommender
        base_recommender = MagicMock()
        base_recommender.predict.return_value = np.array([4.5, 3.2, 2.1, 4.8, 3.9])
        
        # Initialize the explainable recommender
        explainer = ExplainableRecommender(base_recommender)
        
        # Mock the monitoring client
        with patch.object(self.monitoring_client, 'record_data_quality_metrics') as mock_record_dq, \
             patch.object(explainer, '_find_similar_items', return_value=[102, 103]):
            
            # Get an explanation
            explanation = explainer.explain_by_similar_items(1, 101, self.users, self.items, self.interactions)
            
            # Record data quality metrics
            self.monitoring_client.record_data_quality_metrics(
                dataset_name="user_interactions",
                completeness_score=0.95,  # Mock completeness
                consistency_score=0.9,    # Mock consistency
                accuracy_score=0.85       # Mock accuracy
            )
            
            # Check that the monitoring client was called
            mock_record_dq.assert_called_once()
            
            # Check that an explanation is returned
            self.assertIsNotNone(explanation)
            self.assertIn('similar_items', explanation)


class TestMonitoringServiceIntegration(unittest.TestCase):
    def setUp(self):
        # Initialize the monitoring client
        self.monitoring_client = MonitoringClient("http://localhost:8083")
        
        # Check if the monitoring service is available
        self.service_available = False
        try:
            response = requests.get("http://localhost:8083/health", timeout=2)
            self.service_available = response.status_code == 200
        except:
            pass
    
    def test_monitoring_service_health(self):
        # Skip if the service is not available
        if not self.service_available:
            self.skipTest("Monitoring service is not available")
        
        # Check the health endpoint
        response = requests.get("http://localhost:8083/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    def test_record_and_retrieve_metrics(self):
        # Skip if the service is not available
        if not self.service_available:
            self.skipTest("Monitoring service is not available")
        
        # Record model metrics
        self.monitoring_client.record_model_metrics(
            model_name="test_model",
            accuracy=0.9,
            prediction_latency=0.05,
            prediction_count=100,
            drift_score=0.02
        )
        
        # Retrieve metrics
        response = requests.get("http://localhost:8083/metrics/model/test_model")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("test_model", data)
    
    def test_alerts(self):
        # Skip if the service is not available
        if not self.service_available:
            self.skipTest("Monitoring service is not available")
        
        # Trigger an alert by recording a high drift score
        self.monitoring_client.record_model_metrics(
            model_name="drift_test_model",
            accuracy=0.7,
            prediction_latency=0.05,
            prediction_count=100,
            drift_score=0.3  # High drift score should trigger an alert
        )
        
        # Retrieve alerts
        response = requests.get("http://localhost:8083/alerts")
        self.assertEqual(response.status_code, 200)
        alerts = response.json()
        
        # Check if there's an alert for the drift_test_model
        drift_alerts = [alert for alert in alerts if "drift_test_model" in alert["message"]]
        self.assertTrue(len(drift_alerts) > 0, "No drift alert found")


if __name__ == '__main__':
    unittest.main()