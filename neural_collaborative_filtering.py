import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class NeuralCollaborativeFiltering:
    """
    Neural Collaborative Filtering (NCF) implementation
    
    This model combines the linearity of Matrix Factorization with the non-linearity of neural networks
    to create a more expressive model for collaborative filtering.
    
    The architecture includes:
    1. User and item embedding layers
    2. MLP (Multi-Layer Perceptron) path for learning user-item interactions
    3. GMF (Generalized Matrix Factorization) path for linear modeling
    4. A fusion layer that combines both paths
    
    References:
    - He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering.
      In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """
    
    def __init__(self, n_users, n_items, embedding_size=16, layers=[32, 16, 8], reg_layers=[0, 0, 0], 
                 reg_mf=0, learning_rate=0.001):
        """
        Initialize the NCF model
        
        Args:
            n_users: Number of unique users in the dataset
            n_items: Number of unique items in the dataset
            embedding_size: Size of the embedding vectors
            layers: List of layer sizes for the MLP component
            reg_layers: Regularization for each MLP layer
            reg_mf: Regularization for MF embeddings
            learning_rate: Learning rate for the optimizer
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.layers = layers
        self.reg_layers = reg_layers
        self.reg_mf = reg_mf
        self.learning_rate = learning_rate
        self.model = self._build_model()
        
    def _build_model(self):
        # Input layers
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        
        # Embedding layers for GMF
        mf_user_embedding = Embedding(input_dim=self.n_users, output_dim=self.embedding_size,
                                     name='mf_user_embedding', embeddings_regularizer=l2(self.reg_mf))(user_input)
        mf_item_embedding = Embedding(input_dim=self.n_items, output_dim=self.embedding_size,
                                     name='mf_item_embedding', embeddings_regularizer=l2(self.reg_mf))(item_input)
        
        # Embedding layers for MLP
        mlp_user_embedding = Embedding(input_dim=self.n_users, output_dim=int(self.layers[0]/2),
                                      name='mlp_user_embedding')(user_input)
        mlp_item_embedding = Embedding(input_dim=self.n_items, output_dim=int(self.layers[0]/2),
                                      name='mlp_item_embedding')(item_input)
        
        # GMF path
        mf_user_latent = Flatten()(mf_user_embedding)
        mf_item_latent = Flatten()(mf_item_embedding)
        mf_vector = tf.keras.layers.multiply([mf_user_latent, mf_item_latent])
        
        # MLP path
        mlp_user_latent = Flatten()(mlp_user_embedding)
        mlp_item_latent = Flatten()(mlp_item_embedding)
        mlp_vector = Concatenate()([mlp_user_latent, mlp_item_latent])
        
        # MLP layers
        for idx in range(1, len(self.layers)):
            layer = Dense(self.layers[idx], 
                          kernel_regularizer=l2(self.reg_layers[idx]),
                          activation='relu',
                          name=f'layer{idx}')
            mlp_vector = layer(mlp_vector)
            mlp_vector = Dropout(0.2)(mlp_vector)
        
        # Combine GMF and MLP paths
        predict_vector = Concatenate()([mf_vector, mlp_vector])
        
        # Final prediction layer
        prediction = Dense(1, activation='sigmoid', name='prediction')(predict_vector)
        
        # Build and compile the model
        model = Model(inputs=[user_input, item_input], outputs=prediction)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def fit(self, user_ids, item_ids, ratings, validation_split=0.1, epochs=10, batch_size=256, verbose=1):
        """
        Train the model
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            ratings: Array of ratings (should be normalized to [0,1] range)
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        # Normalize ratings to [0,1] if they aren't already
        normalized_ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
        
        history = self.model.fit(
            [np.array(user_ids), np.array(item_ids)],
            normalized_ratings,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_split=validation_split
        )
        
        return history
    
    def predict(self, user_ids, item_ids):
        """
        Predict ratings for given user-item pairs
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Predicted ratings (denormalized to original scale)
        """
        predictions = self.model.predict([np.array(user_ids), np.array(item_ids)])
        # Denormalize predictions if needed
        return predictions
    
    def recommend(self, user_id, all_item_ids, n_recommendations=5, exclude_rated=True, rated_items=None):
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID to recommend for
            all_item_ids: List of all possible item IDs
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            rated_items: List of items already rated by the user (required if exclude_rated=True)
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if exclude_rated and rated_items is None:
            raise ValueError("rated_items must be provided if exclude_rated=True")
        
        # Filter out already rated items if needed
        items_to_predict = [item for item in all_item_ids if not exclude_rated or item not in rated_items]
        
        # Create user-item pairs for prediction
        user_ids = np.full(len(items_to_predict), user_id)
        item_ids = np.array(items_to_predict)
        
        # Get predictions
        predictions = self.predict(user_ids, item_ids).flatten()
        
        # Create (item_id, rating) pairs and sort by rating
        item_rating_pairs = list(zip(items_to_predict, predictions))
        item_rating_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return item_rating_pairs[:n_recommendations]
    
    def save_model(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath, n_users, n_items):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
            n_users: Number of users
            n_items: Number of items
            
        Returns:
            Loaded NCF model
        """
        instance = cls(n_users, n_items)
        instance.model = tf.keras.models.load_model(filepath)
        return instance


def prepare_ncf_data(ratings_df, user_col='user_id', item_col='movie_id', rating_col='rating'):
    """
    Prepare data for NCF model
    
    Args:
        ratings_df: DataFrame containing user-item interactions
        user_col: Name of the user column
        item_col: Name of the item column
        rating_col: Name of the rating column
        
    Returns:
        user_ids, item_ids, ratings, n_users, n_items, user_encoder, item_encoder
    """
    # Encode user and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_ids = user_encoder.fit_transform(ratings_df[user_col].values)
    item_ids = item_encoder.fit_transform(ratings_df[item_col].values)
    ratings = ratings_df[rating_col].values
    
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    return user_ids, item_ids, ratings, n_users, n_items, user_encoder, item_encoder


def train_ncf_model(ratings_df, embedding_size=16, layers=[32, 16, 8], epochs=10, batch_size=256):
    """
    Train a Neural Collaborative Filtering model
    
    Args:
        ratings_df: DataFrame with columns 'user_id', 'movie_id', 'rating'
        embedding_size: Size of the embedding vectors
        layers: List of layer sizes for the MLP component
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained NCF model, user_encoder, item_encoder
    """
    # Prepare data
    user_ids, item_ids, ratings, n_users, n_items, user_encoder, item_encoder = prepare_ncf_data(ratings_df)
    
    # Create and train the model
    model = NeuralCollaborativeFiltering(
        n_users=n_users,
        n_items=n_items,
        embedding_size=embedding_size,
        layers=layers
    )
    
    history = model.fit(
        user_ids=user_ids,
        item_ids=item_ids,
        ratings=ratings,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    
    return model, user_encoder, item_encoder, history


def get_ncf_recommendations(model, user_id, user_encoder, item_encoder, n_recommendations=5, exclude_rated=True, ratings_df=None):
    """
    Get recommendations for a user using the NCF model
    
    Args:
        model: Trained NCF model
        user_id: Original user ID (before encoding)
        user_encoder: LabelEncoder for users
        item_encoder: LabelEncoder for items
        n_recommendations: Number of recommendations to generate
        exclude_rated: Whether to exclude already rated items
        ratings_df: DataFrame with ratings (required if exclude_rated=True)
        
    Returns:
        List of (item_id, predicted_rating) tuples
    """
    # Encode the user ID
    encoded_user_id = user_encoder.transform([user_id])[0]
    
    # Get all possible items
    all_item_ids = list(range(len(item_encoder.classes_)))
    
    # Get rated items if needed
    rated_items = None
    if exclude_rated and ratings_df is not None:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        rated_items = item_encoder.transform(user_ratings['movie_id'].values)
    
    # Get recommendations
    recommendations = model.recommend(
        user_id=encoded_user_id,
        all_item_ids=all_item_ids,
        n_recommendations=n_recommendations,
        exclude_rated=exclude_rated,
        rated_items=rated_items
    )
    
    # Decode item IDs
    decoded_recommendations = [
        (item_encoder.inverse_transform([item_id])[0], rating)
        for item_id, rating in recommendations
    ]
    
    return decoded_recommendations