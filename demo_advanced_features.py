import pandas as pd
import numpy as np
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the advanced features modules
from microservices.advanced_features.multi_modal_recommender import MultiModalRecommender, create_multi_modal_recommender
from microservices.advanced_features.reinforcement_learning import create_rl_recommender, get_rl_recommendations
from microservices.advanced_features.explainable_ai import create_explainable_recommender, get_explanation

# Import the monitoring client
from microservices.monitoring.monitoring_client import MonitoringClient


def load_sample_data():
    """Load sample data for demonstration"""
    # Create sample users
    users = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 70, 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'occupation': np.random.choice(['student', 'engineer', 'teacher', 'doctor', 'other'], 100)
    })
    
    # Create sample items
    categories = ['electronics', 'books', 'clothing', 'home', 'sports', 'beauty', 'toys', 'food', 'health', 'automotive']
    items = pd.DataFrame({
        'item_id': range(101, 301),
        'category': np.random.choice(categories, 200),
        'price': np.random.uniform(10, 500, 200),
        'brand': np.random.choice(['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE'], 200),
        'rating': np.random.uniform(1, 5, 200).round(1)
    })
    
    # Create sample interactions
    num_interactions = 1000
    interactions = pd.DataFrame({
        'user_id': np.random.choice(users['user_id'], num_interactions),
        'item_id': np.random.choice(items['item_id'], num_interactions),
        'rating': np.random.uniform(1, 5, num_interactions).round(1),
        'timestamp': pd.date_range(start='2023-01-01', periods=num_interactions, freq='H')
    })
    
    return users, items, interactions


def demo_multi_modal_recommender(users, items, interactions, monitoring_client=None):
    """Demonstrate the multi-modal recommender"""
    print("\n" + "=" * 80)
    print("MULTI-MODAL RECOMMENDER DEMONSTRATION")
    print("=" * 80)
    
    # Create a multi-modal recommender
    print("\nCreating and training multi-modal recommender...")
    recommender = create_multi_modal_recommender()
    
    # Train the recommender
    start_time = time.time()
    recommender.train(users, items, interactions)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get recommendations for a few users
    test_users = users.sample(3)['user_id'].tolist()
    
    print("\nGenerating recommendations for sample users:")
    for user_id in test_users:
        start_time = time.time()
        recommendations = recommender.get_recommendations(user_id, users, items, top_n=5)
        prediction_time = time.time() - start_time
        
        # Record metrics if monitoring client is available
        if monitoring_client:
            monitoring_client.record_model_metrics(
                model_type="multi_modal_recommender",
                accuracy=0.02,  # Mock accuracy
                latency=0.1,    # Mock latency
                throughput=1    # Mock throughput
            )
        
        # Display recommendations
        print(f"\nRecommendations for User {user_id}:")
        if len(recommendations) > 0:
            rec_df = pd.merge(recommendations, items, on='item_id')
            print(tabulate(rec_df[['item_id', 'category', 'price', 'predicted_rating']], 
                          headers='keys', tablefmt='pretty', showindex=False))
        else:
            print("No recommendations found.")
    
    return recommender


def demo_reinforcement_learning(users, items, interactions, monitoring_client=None):
    """Demonstrate the reinforcement learning recommender"""
    print("\n" + "=" * 80)
    print("REINFORCEMENT LEARNING RECOMMENDER DEMONSTRATION")
    print("=" * 80)
    
    # Create an RL recommender
    print("\nCreating and training reinforcement learning recommender...")
    # Pass interactions as ratings data with user/item features
    recommender = create_rl_recommender(
        ratings_df=interactions,
        item_features_df=items,
        user_features_df=users
    )
    
    # Train the recommender
    start_time = time.time()
    recommender.train(n_episodes=10)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Get recommendations for a few users
    test_users = users.sample(3)['user_id'].tolist()
    
    print("\nGenerating recommendations for sample users:")
    for user_id in test_users:
        start_time = time.time()
        recommendations = get_rl_recommendations(recommender, user_id, items, top_n=5)
        prediction_time = time.time() - start_time
        
        # Record user experience metrics if monitoring client is available
        if monitoring_client:
            # Calculate mock diversity and coverage scores
            if len(recommendations) > 0:
                categories = items.loc[items['item_id'].isin(recommendations['item_id']), 'category'].unique()
                diversity_score = len(categories) / len(recommendations)
                coverage_score = len(recommendations) / len(items)
            else:
                diversity_score = 0
                coverage_score = 0
                
            monitoring_client.record_user_experience_metrics(
                user_id=user_id,
                satisfaction_score=4.2,  # Mock satisfaction
                diversity_score=diversity_score,
                coverage_score=coverage_score
            )
        
        # Display recommendations
        print(f"\nRecommendations for User {user_id}:")
        if len(recommendations) > 0:
            rec_df = pd.merge(recommendations, items, on='item_id')
            print(tabulate(rec_df[['item_id', 'category', 'price', 'reward']], 
                          headers='keys', tablefmt='pretty', showindex=False))
        else:
            print("No recommendations found.")
    
    return recommender


def demo_explainable_ai(users, items, interactions, base_recommender, monitoring_client=None):
    """Demonstrate the explainable AI features"""
    print("\n" + "=" * 80)
    print("EXPLAINABLE AI DEMONSTRATION")
    print("=" * 80)
    
    # Create an explainable recommender
    print("\nCreating explainable recommender...")
    explainer = create_explainable_recommender(base_recommender)
    
    # Get recommendations for a user
    user_id = users.sample(1)['user_id'].iloc[0]
    recommendations = base_recommender.get_recommendations(user_id, users, items, top_n=5)
    
    if len(recommendations) == 0:
        print("No recommendations found for explanation.")
        return
    
    # Get the first recommended item
    item_id = recommendations.iloc[0]['item_id']
    
    print(f"\nExplaining recommendation of Item {item_id} for User {user_id}:")
    
    # Get different types of explanations
    start_time = time.time()
    
    # Similar items explanation
    similar_items_explanation = get_explanation(explainer, user_id, item_id, users, items, interactions, 
                                              explanation_type='similar_items')
    
    # User history explanation
    user_history_explanation = get_explanation(explainer, user_id, item_id, users, items, interactions, 
                                             explanation_type='user_history')
    
    # Feature contribution explanation
    feature_explanation = get_explanation(explainer, user_id, item_id, users, items, interactions, 
                                        explanation_type='features')
    
    # Natural language explanation
    nl_explanation = explainer.get_natural_language_explanation(user_id, item_id, users, items, interactions)
    
    explanation_time = time.time() - start_time
    
    # Record data quality metrics if monitoring client is available
    if monitoring_client:
        monitoring_client.record_data_quality_metrics(
            dataset_name="user_interactions",
            completeness_score=0.95,  # Mock completeness
            consistency_score=0.9,    # Mock consistency
            accuracy_score=0.85       # Mock accuracy
        )
    
    # Display explanations
    print("\nSimilar Items Explanation:")
    if 'similar_items' in similar_items_explanation:
        similar_item_ids = similar_items_explanation['similar_items']
        similar_items_df = items[items['item_id'].isin(similar_item_ids)]
        print(tabulate(similar_items_df[['item_id', 'category', 'price', 'rating']], 
                      headers='keys', tablefmt='pretty', showindex=False))
    else:
        print("No similar items found.")
    
    print("\nUser History Explanation:")
    if 'user_history' in user_history_explanation:
        history_item_ids = user_history_explanation['user_history']
        history_df = items[items['item_id'].isin(history_item_ids)]
        print(tabulate(history_df[['item_id', 'category', 'price', 'rating']], 
                      headers='keys', tablefmt='pretty', showindex=False))
    else:
        print("No user history found.")
    
    print("\nFeature Contribution Explanation:")
    if 'feature_importance' in feature_explanation:
        for feature, importance in feature_explanation['feature_importance'].items():
            print(f"{feature}: {importance:.4f}")
    else:
        print("No feature importance information available.")
    
    print("\nNatural Language Explanation:")
    print(nl_explanation)
    
    print(f"\nExplanations generated in {explanation_time:.2f} seconds")


def demo_monitoring_dashboard(monitoring_client):
    """Demonstrate the monitoring dashboard"""
    print("\n" + "=" * 80)
    print("MONITORING DASHBOARD DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Get the dashboard summary
        dashboard = monitoring_client.get_dashboard_summary()
        
        print("\nSystem Metrics:")
        system_metrics = dashboard.get('system_metrics', {})
        for metric, value in system_metrics.items():
            print(f"{metric}: {value}")
        
        print("\nService Health:")
        service_health = dashboard.get('service_health', {})
        for service, status in service_health.items():
            print(f"{service}: {status}")
        
        print("\nModel Performance:")
        model_performance = dashboard.get('model_performance', {})
        for model, metrics in model_performance.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        
        print("\nData Quality:")
        data_quality = dashboard.get('data_quality', {})
        for dataset, metrics in data_quality.items():
            print(f"\n{dataset}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
        
        print("\nUser Experience:")
        user_experience = dashboard.get('user_experience', {})
        for metric, value in user_experience.items():
            print(f"{metric}: {value}")
        
        print("\nActive Alerts:")
        alerts = monitoring_client.get_alerts()
        if alerts:
            for alert in alerts:
                print(f"[{alert['severity']}] {alert['message']} - {alert['timestamp']}")
        else:
            print("No active alerts")
            
    except Exception as e:
        print(f"Error accessing monitoring dashboard: {str(e)}")
        print("Note: This is expected if the monitoring service is not running.")


def main():
    parser = argparse.ArgumentParser(description='Demonstration of Advanced Features and Monitoring')
    parser.add_argument('--monitoring-url', type=str, default='http://localhost:8083',
                        help='URL of the monitoring service')
    parser.add_argument('--skip-monitoring', action='store_true',
                        help='Skip monitoring integration')
    args = parser.parse_args()
    
    # Initialize monitoring client if not skipped
    monitoring_client = None
    if not args.skip_monitoring:
        monitoring_client = MonitoringClient(args.monitoring_url)
    
    # Load sample data
    print("Loading sample data...")
    users, items, interactions = load_sample_data()
    print(f"Loaded {len(users)} users, {len(items)} items, and {len(interactions)} interactions")
    
    # Demonstrate multi-modal recommender
    multi_modal_recommender = demo_multi_modal_recommender(users, items, interactions, monitoring_client)
    
    # Demonstrate reinforcement learning
    recommender = demo_reinforcement_learning(users, items, interactions, monitoring_client)
    
    # Demonstrate explainable AI
    demo_explainable_ai(users, items, interactions, multi_modal_recommender, monitoring_client)
    
    # Demonstrate monitoring dashboard
    if monitoring_client:
        demo_monitoring_dashboard(monitoring_client)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()