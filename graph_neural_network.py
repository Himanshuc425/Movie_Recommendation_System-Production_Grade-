import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import networkx as nx

class GraphConvolutionLayer(tf.keras.layers.Layer):
    """
    Graph Convolutional Network (GCN) layer implementation
    
    This layer performs the graph convolution operation:
    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    where:
    - A is the adjacency matrix (with self-connections)
    - D is the degree matrix
    - H^(l) is the matrix of activations in the lth layer
    - W^(l) is the weight matrix for the lth layer
    - σ is the activation function
    """
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        
    def build(self, input_shape):
        # input_shape = [features_shape, adjacency_shape]
        features_shape = input_shape[0]
        
        # Create weight matrix
        self.kernel = self.add_weight(
            shape=(features_shape[-1], self.units),
            initializer='glorot_uniform',
            name='kernel')
        
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias')
        
        self.built = True
        
    def call(self, inputs):
        # inputs = [features, normalized_adjacency]
        features, normalized_adjacency = inputs
        
        # Graph convolution operation
        support = tf.matmul(features, self.kernel)
        output = tf.matmul(normalized_adjacency, support)
        output = output + self.bias
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class GNNRecommender:
    """
    Graph Neural Network for Recommendation Systems
    
    This model uses a Graph Convolutional Network (GCN) to learn representations
    of users and items from their interaction graph, and then uses these
    representations to make recommendations.
    
    References:
    - Berg, R. v. d., Kipf, T. N., & Welling, M. (2017). Graph convolutional matrix completion.
      arXiv preprint arXiv:1706.02263.
    - Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). Neural graph collaborative filtering.
      In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development
      in Information Retrieval (pp. 165-174).
    """
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2, learning_rate=0.001):
        """
        Initialize the GNN recommender
        
        Args:
            n_users: Number of users in the dataset
            n_items: Number of items in the dataset
            embedding_dim: Dimension of the embeddings
            n_layers: Number of GCN layers
            learning_rate: Learning rate for the optimizer
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.model = None
        self.user_embedding = None
        self.item_embedding = None
        self.normalized_adjacency = None
        
    def _create_adjacency_matrix(self, user_ids, item_ids):
        """
        Create the adjacency matrix for the user-item interaction graph
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            
        Returns:
            Normalized adjacency matrix as a TensorFlow sparse tensor
        """
        # Create a bipartite graph adjacency matrix
        # [0, n_users-1] are user nodes, [n_users, n_users+n_items-1] are item nodes
        n = self.n_users + self.n_items
        
        # Create edges: user-to-item and item-to-user
        edge_list = []
        for u, i in zip(user_ids, item_ids):
            # User -> Item edge
            edge_list.append((u, i + self.n_users))
            # Item -> User edge (for undirected graph)
            edge_list.append((i + self.n_users, u))
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edge_list)
        
        # Add self-loops
        for i in range(n):
            G.add_edge(i, i)
        
        # Get adjacency matrix
        A = nx.adjacency_matrix(G).astype(np.float32)
        
        # Normalize adjacency matrix: D^(-1/2) A D^(-1/2)
        rowsum = np.array(A.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adjacency = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
        
        # Convert to TensorFlow sparse tensor
        normalized_adjacency = tf.convert_to_tensor(normalized_adjacency.todense(), dtype=tf.float32)
        
        return normalized_adjacency
    
    def _build_model(self):
        """
        Build the GNN model
        
        Returns:
            Compiled Keras model
        """
        # Input layers
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        
        # Embedding layers
        user_embedding = tf.keras.layers.Embedding(
            input_dim=self.n_users,
            output_dim=self.embedding_dim,
            name='user_embedding'
        )(user_input)
        
        item_embedding = tf.keras.layers.Embedding(
            input_dim=self.n_items,
            output_dim=self.embedding_dim,
            name='item_embedding'
        )(item_input)
        
        # Flatten embeddings
        user_embedding_flat = tf.keras.layers.Flatten()(user_embedding)
        item_embedding_flat = tf.keras.layers.Flatten()(item_embedding)
        
        # Concatenate user and item embeddings
        concat_embedding = tf.keras.layers.Concatenate()([user_embedding_flat, item_embedding_flat])
        
        # Dense layers
        x = Dense(64, activation='relu')(concat_embedding)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='prediction')(x)
        
        # Create model
        model = Model(inputs=[user_input, item_input], outputs=output)
        
        # Compile model
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        
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
        # Create the adjacency matrix
        self.normalized_adjacency = self._create_adjacency_matrix(user_ids, item_ids)
        
        # Build the model if it doesn't exist
        if self.model is None:
            self.model = self._build_model()
        
        # Normalize ratings to [0,1] if they aren't already
        normalized_ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
        
        # Train the model
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
            Predicted ratings
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions = self.model.predict([np.array(user_ids), np.array(item_ids)])
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
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
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
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath, n_users, n_items, embedding_dim=64):
        """
        Load a saved model
        
        Args:
            filepath: Path to the saved model
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of the embeddings
            
        Returns:
            Loaded GNN model
        """
        instance = cls(n_users, n_items, embedding_dim)
        instance.model = tf.keras.models.load_model(filepath)
        return instance


def prepare_gnn_data(ratings_df, user_col='user_id', item_col='movie_id', rating_col='rating'):
    """
    Prepare data for GNN model
    
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


def train_gnn_model(ratings_df, embedding_dim=64, n_layers=2, epochs=10, batch_size=256):
    """
    Train a Graph Neural Network model for recommendations
    
    Args:
        ratings_df: DataFrame with columns 'user_id', 'movie_id', 'rating'
        embedding_dim: Dimension of the embeddings
        n_layers: Number of GCN layers
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained GNN model, user_encoder, item_encoder
    """
    # Prepare data
    user_ids, item_ids, ratings, n_users, n_items, user_encoder, item_encoder = prepare_gnn_data(ratings_df)
    
    # Create and train the model
    model = GNNRecommender(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        n_layers=n_layers
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


def get_gnn_recommendations(model, user_id, user_encoder, item_encoder, n_recommendations=5, exclude_rated=True, ratings_df=None):
    """
    Get recommendations for a user using the GNN model
    
    Args:
        model: Trained GNN model
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