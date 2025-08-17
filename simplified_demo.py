import numpy as np
import pandas as pd
import tensorflow as tf

# Import the implemented modules
from neural_collaborative_filtering import NeuralCollaborativeFiltering
from graph_neural_network import GNNRecommender
from enhanced_features import EnhancedFeatureEngineering

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_sample_data(n_users=100, n_items=50, n_ratings=500):
    """
    Create a small sample dataset for demonstration purposes
    
    Args:
        n_users: Number of users
        n_items: Number of items (movies)
        n_ratings: Number of ratings
        
    Returns:
        movies_df: DataFrame with movie features
        ratings_df: DataFrame with user-movie ratings
    """
    print("Creating sample data...")
    
    # Create movie data
    movies_data = {
        'movie_id': list(range(n_items)),
        'title': [f'Movie {i}' for i in range(n_items)],
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Thriller'], n_items),
        'year': np.random.randint(1990, 2023, n_items),
        'popularity': np.random.uniform(1, 10, n_items)
    }
    movies_df = pd.DataFrame(movies_data)
    
    # Create ratings data
    user_ids = np.random.randint(0, n_users, n_ratings)
    movie_ids = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.uniform(1, 5, n_ratings)
    
    ratings_data = {
        'user_id': user_ids,
        'movie_id': movie_ids,
        'rating': ratings,
        'timestamp': pd.date_range(start='2020-01-01', periods=n_ratings, freq='H')
    }
    ratings_df = pd.DataFrame(ratings_data).drop_duplicates(['user_id', 'movie_id'])
    
    print(f"Created {len(movies_df)} movies and {len(ratings_df)} ratings")
    return movies_df, ratings_df

def demonstrate_neural_collaborative_filtering(ratings_df):
    """
    Demonstrate how to use the Neural Collaborative Filtering model
    
    Args:
        ratings_df: DataFrame with user-movie ratings
    """
    print("\n" + "="*60)
    print("NEURAL COLLABORATIVE FILTERING DEMONSTRATION")
    print("="*60)
    
    # Get unique users and items
    n_users = ratings_df['user_id'].nunique()
    n_items = ratings_df['movie_id'].nunique()
    
    print(f"Building NCF model with {n_users} users and {n_items} items")
    
    # Create the NCF model
    ncf_model = NeuralCollaborativeFiltering(
        n_users=n_users,
        n_items=n_items,
        embedding_size=8,  # Smaller for demonstration
        layers=[16, 8]     # Smaller for demonstration
    )
    
    # Show model architecture
    print("\nNeural Collaborative Filtering Model Architecture:")
    print("- User and Item Embedding Layers")
    print("- Generalized Matrix Factorization (GMF) Path")
    print("- Multi-Layer Perceptron (MLP) Path")
    print("- Fusion Layer")
    
    print("\nModel would be trained with:")
    print("- User IDs: Input to user embedding layer")
    print("- Item IDs: Input to item embedding layer")
    print("- Ratings: Target values for prediction")
    
    print("\nAfter training, the model can generate personalized recommendations")
    print("by predicting ratings for user-item pairs not in the training set.")

def demonstrate_graph_neural_network(ratings_df):
    """
    Demonstrate how to use the Graph Neural Network model
    
    Args:
        ratings_df: DataFrame with user-movie ratings
    """
    print("\n" + "="*60)
    print("GRAPH NEURAL NETWORK DEMONSTRATION")
    print("="*60)
    
    # Get unique users and items
    n_users = ratings_df['user_id'].nunique()
    n_items = ratings_df['movie_id'].nunique()
    
    print(f"Building GNN model with {n_users} users and {n_items} items")
    
    # Create the GNN model
    gnn_model = GNNRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=8,  # Smaller for demonstration
        n_layers=1        # Fewer layers for demonstration
    )
    
    # Show model architecture
    print("\nGraph Neural Network Model Architecture:")
    print("- User-Item Interaction Graph Construction")
    print("- Graph Convolutional Layers")
    print("- Message Passing Between Nodes")
    print("- Node Embedding Generation")
    print("- Rating Prediction from Embeddings")
    
    print("\nModel would be trained with:")
    print("- User-Item Interaction Graph: Built from ratings data")
    print("- Node Features: Initial embeddings for users and items")
    print("- Ratings: Target values for prediction")
    
    print("\nAfter training, the model can generate recommendations")
    print("by leveraging the graph structure of user-item interactions.")

def demonstrate_enhanced_features(movies_df, ratings_df):
    """
    Demonstrate how to use the enhanced feature engineering
    
    Args:
        movies_df: DataFrame with movie features
        ratings_df: DataFrame with user-movie ratings
    """
    print("\n" + "="*60)
    print("ENHANCED FEATURE ENGINEERING DEMONSTRATION")
    print("="*60)
    
    # Create feature engineering pipeline
    feature_engineering = EnhancedFeatureEngineering(
        user_col='user_id',
        item_col='movie_id',
        rating_col='rating',
        timestamp_col='timestamp',
        genre_col='genre',
        text_cols=['title']
    )
    
    # Show available feature types
    print("\nEnhanced Feature Types:")
    print("1. Temporal Features:")
    print("   - Time of day, day of week, month, year")
    print("   - Cyclical encoding of time features")
    print("   - Recency features (days since last interaction)")
    
    print("\n2. User Context Features:")
    print("   - User rating patterns (avg, std, count)")
    print("   - User activity level")
    print("   - User preferences")
    
    print("\n3. Item Context Features:")
    print("   - Item popularity")
    print("   - Item category/genre encoding")
    print("   - Item rating patterns")
    
    print("\n4. Interaction Context Features:")
    print("   - User-item affinity scores")
    print("   - Sequential patterns")
    print("   - Cross-feature interactions")
    
    print("\nThese enhanced features can significantly improve recommendation quality")
    print("by capturing temporal patterns, user/item context, and interaction dynamics.")

def main():
    """
    Main function to demonstrate the advanced recommendation system components
    """
    print("\n" + "="*60)
    print("ADVANCED MOVIE RECOMMENDATION SYSTEM COMPONENTS")
    print("="*60)
    
    # Create sample data
    movies_df, ratings_df = create_sample_data()
    
    # Demonstrate Neural Collaborative Filtering
    demonstrate_neural_collaborative_filtering(ratings_df)
    
    # Demonstrate Graph Neural Network
    demonstrate_graph_neural_network(ratings_df)
    
    # Demonstrate Enhanced Feature Engineering
    demonstrate_enhanced_features(movies_df, ratings_df)
    
    print("\n" + "="*60)
    print("PHASE 1: ADVANCED ALGORITHM ARCHITECTURE COMPLETED")
    print("="*60)
    print("\nThe implementation includes:")
    print("1. Neural Collaborative Filtering (NCF) model")
    print("2. Graph Neural Network (GNN) model")
    print("3. Enhanced feature engineering with temporal and contextual features")
    
    print("\nTo use these components in your project:")
    print("1. Import the modules: neural_collaborative_filtering.py, graph_neural_network.py, enhanced_features.py")
    print("2. Prepare your data using the provided functions")
    print("3. Train the models and generate recommendations")
    
    print("\nRefer to the documentation in each module for detailed usage instructions.")

if __name__ == "__main__":
    main()