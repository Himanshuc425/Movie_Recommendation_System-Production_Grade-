import unittest
import pandas as pd
import numpy as np
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the advanced features modules
from microservices.advanced_features.multi_modal_recommender import MultiModalRecommender
from microservices.advanced_features.reinforcement_learning import RLRecommender, RecommenderEnvironment, DQNAgent
from microservices.advanced_features.explainable_ai import ExplainableRecommender, ExplainableNCF, ExplainableGNN

class TestMultiModalRecommender(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.users = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'age': [25, 30, 35, 40, 45],
            'gender': ['M', 'F', 'M', 'F', 'M']
        })
        
        self.items = pd.DataFrame({
            'item_id': [101, 102, 103, 104, 105],
            'category': ['electronics', 'books', 'clothing', 'home', 'sports'],
            'price': [100, 20, 50, 200, 75]
        })
        
        self.interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 4, 5],
            'item_id': [101, 103, 102, 104, 105, 101, 102],
            'rating': [5, 4, 3, 5, 4, 2, 3]
        })
        
        # Mock image and text features
        self.image_features = {item_id: np.random.rand(10) for item_id in self.items['item_id']}
        self.text_features = {item_id: np.random.rand(10) for item_id in self.items['item_id']}
        
        # Initialize the recommender with mocked features
        with patch('microservices.advanced_features.multi_modal_recommender.MultiModalRecommender._load_image_features', 
                  return_value=self.image_features), \
             patch('microservices.advanced_features.multi_modal_recommender.MultiModalRecommender._load_text_features', 
                  return_value=self.text_features):
            self.recommender = MultiModalRecommender()
    
    def test_initialization(self):
        self.assertIsNotNone(self.recommender)
        self.assertIsNotNone(self.recommender.image_features)
        self.assertIsNotNone(self.recommender.text_features)
    
    def test_prepare_data(self):
        with patch('microservices.advanced_features.multi_modal_recommender.MultiModalRecommender._load_image_features', 
                  return_value=self.image_features), \
             patch('microservices.advanced_features.multi_modal_recommender.MultiModalRecommender._load_text_features', 
                  return_value=self.text_features):
            X, y = self.recommender.prepare_data(self.users, self.items, self.interactions)
            
            # Check that the data has been prepared correctly
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertEqual(len(X), len(self.interactions))
            self.assertEqual(len(y), len(self.interactions))
    
    def test_get_recommendations(self):
        # Mock the model prediction
        with patch.object(self.recommender, 'model', MagicMock()):
            self.recommender.model.predict.return_value = np.array([4.5, 3.2, 2.1, 4.8, 3.9])
            
            # Get recommendations for a user
            recommendations = self.recommender.get_recommendations(1, self.users, self.items, top_n=3)
            
            # Check that recommendations are returned
            self.assertIsNotNone(recommendations)
            self.assertLessEqual(len(recommendations), 3)


class TestReinforcementLearning(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M']
        })
        
        self.items = pd.DataFrame({
            'item_id': [101, 102, 103, 104, 105],
            'category': ['electronics', 'books', 'clothing', 'home', 'sports'],
            'price': [100, 20, 50, 200, 75]
        })
        
        self.interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [101, 103, 102, 104, 105],
            'rating': [5, 4, 3, 5, 4]
        })
        
        # Initialize the environment and agent
        self.env = RecommenderEnvironment(self.users, self.items, self.interactions)
        
        # Mock the DQN agent
        self.agent = MagicMock(spec=DQNAgent)
        self.agent.act.return_value = 0  # Always choose the first action
        
        # Initialize the recommender
        self.recommender = RLRecommender(self.env, self.agent)
    
    def test_initialization(self):
        self.assertIsNotNone(self.recommender)
        self.assertIsNotNone(self.recommender.env)
        self.assertIsNotNone(self.recommender.agent)
    
    def test_train(self):
        # Mock the training process
        with patch.object(self.agent, 'train', return_value=None):
            self.recommender.train(episodes=5)
            # Check that the agent's train method was called
            self.agent.train.assert_called_once()
    
    def test_get_recommendations(self):
        # Mock the agent's act method
        self.agent.act.return_value = 0  # Always choose the first action
        
        # Get recommendations for a user
        recommendations = self.recommender.get_recommendations(1, top_n=3)
        
        # Check that recommendations are returned
        self.assertIsNotNone(recommendations)
        self.assertLessEqual(len(recommendations), 3)


class TestExplainableAI(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.users = pd.DataFrame({
            'user_id': [1, 2, 3],
            'age': [25, 30, 35],
            'gender': ['M', 'F', 'M']
        })
        
        self.items = pd.DataFrame({
            'item_id': [101, 102, 103, 104, 105],
            'category': ['electronics', 'books', 'clothing', 'home', 'sports'],
            'price': [100, 20, 50, 200, 75]
        })
        
        self.interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'item_id': [101, 103, 102, 104, 105],
            'rating': [5, 4, 3, 5, 4]
        })
        
        # Mock the base recommender
        self.base_recommender = MagicMock()
        self.base_recommender.predict.return_value = np.array([4.5, 3.2, 2.1, 4.8, 3.9])
        
        # Initialize the explainable recommender
        self.explainer = ExplainableRecommender(self.base_recommender)
    
    def test_initialization(self):
        self.assertIsNotNone(self.explainer)
        self.assertIsNotNone(self.explainer.model)
    
    def test_explain_by_similar_items(self):
        # Mock the similar items calculation
        with patch.object(self.explainer, '_find_similar_items', return_value=[102, 103]):
            explanation = self.explainer.explain_by_similar_items(1, 101, self.users, self.items, self.interactions)
            
            # Check that an explanation is returned
            self.assertIsNotNone(explanation)
            self.assertIn('similar_items', explanation)
    
    def test_explain_by_features(self):
        # Mock the SHAP explainer
        with patch('microservices.advanced_features.explainable_ai.shap.Explainer'):
            explanation = self.explainer.explain_by_features(1, 101, self.users, self.items, self.interactions)
            
            # Check that an explanation is returned
            self.assertIsNotNone(explanation)
            self.assertIn('feature_importance', explanation)
    
    def test_get_natural_language_explanation(self):
        # Get a natural language explanation
        explanation = self.explainer.get_natural_language_explanation(1, 101, self.users, self.items, self.interactions)
        
        # Check that an explanation is returned
        self.assertIsNotNone(explanation)
        self.assertIsInstance(explanation, str)


def calculate_novelty(recommendations):
    # Implement novelty metric

    if __name__ == '__main__':
        unittest.main()
    