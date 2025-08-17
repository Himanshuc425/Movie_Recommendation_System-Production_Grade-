import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

class MultiModalRecommender:
    """
    Multi-Modal Recommendation System that combines user-item interactions with
    image and text content for more accurate and diverse recommendations.
    
    This model integrates:
    1. Collaborative filtering signals (user-item interactions)
    2. Visual features from movie posters/thumbnails
    3. Textual features from movie descriptions/reviews
    
    The model uses a hybrid architecture that fuses these different modalities
    to generate recommendations that leverage both interaction patterns and content.
    """
    
    def __init__(self, 
                 n_users, 
                 n_items, 
                 embedding_size=32,
                 text_input_length=200,
                 text_vocab_size=10000,
                 text_embedding_dim=100,
                 image_input_shape=(32, 32, 3),
                 learning_rate=0.001,
                 reg_factor=0.0001):
        """
        Initialize the multi-modal recommender model
        
        Args:
            n_users: Number of users in the dataset
            n_items: Number of items in the dataset
            embedding_size: Size of user and item embeddings for collaborative filtering
            text_input_length: Maximum length of text input sequences
            text_vocab_size: Size of text vocabulary
            text_embedding_dim: Dimension of text embeddings
            image_input_shape: Shape of input images (height, width, channels)
            learning_rate: Learning rate for optimizer
            reg_factor: Regularization factor
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.text_input_length = text_input_length
        self.text_vocab_size = text_vocab_size
        self.text_embedding_dim = text_embedding_dim
        self.image_input_shape = image_input_shape
        self.learning_rate = learning_rate
        self.reg_factor = reg_factor
        
        # Initialize text tokenizer
        self.tokenizer = None
        
        # Build the model
        self.model = self._build_model()
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
    
    def _build_collaborative_filtering_branch(self, user_input, item_input):
        """
        Build the collaborative filtering branch of the model
        
        Args:
            user_input: User input tensor
            item_input: Item input tensor
            
        Returns:
            CF branch output tensor
        """
        # User embedding
        user_embedding = Embedding(input_dim=self.n_users,
                                  output_dim=self.embedding_size,
                                  embeddings_regularizer=l2(self.reg_factor),
                                  input_length=1,
                                  name='user_embedding')(user_input)
        user_embedding = Flatten()(user_embedding)
        
        # Item embedding
        item_embedding = Embedding(input_dim=self.n_items,
                                  output_dim=self.embedding_size,
                                  embeddings_regularizer=l2(self.reg_factor),
                                  input_length=1,
                                  name='item_embedding')(item_input)
        item_embedding = Flatten()(item_embedding)
        
        # Concatenate embeddings
        cf_vector = Concatenate()([user_embedding, item_embedding])
        cf_vector = Dense(64, activation='relu')(cf_vector)
        cf_vector = Dropout(0.2)(cf_vector)
        cf_vector = Dense(32, activation='relu')(cf_vector)
        
        return cf_vector
    
    def _build_image_branch(self, image_input):
        """
        Build the image processing branch using a simplified CNN for smaller images
        
        Args:
            image_input: Image input tensor
            
        Returns:
            Image branch output tensor
        """
        # For smaller images, use a simpler CNN architecture instead of ResNet50
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        image_vector = Dense(64, activation='relu')(x)
        
        return image_vector
    
    def _build_text_branch(self, text_input):
        """
        Build the text processing branch using embeddings and CNN
        
        Args:
            text_input: Text input tensor
            
        Returns:
            Text branch output tensor
        """
        # Text embedding layer
        text_embedding = Embedding(input_dim=self.text_vocab_size,
                                  output_dim=self.text_embedding_dim,
                                  input_length=self.text_input_length)(text_input)
        
        # Apply 1D convolutions of different sizes to capture n-grams
        conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(text_embedding)
        conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu')(text_embedding)
        conv3 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu')(text_embedding)
        
        # Max pooling
        pool1 = tf.keras.layers.GlobalMaxPooling1D()(conv1)
        pool2 = tf.keras.layers.GlobalMaxPooling1D()(conv2)
        pool3 = tf.keras.layers.GlobalMaxPooling1D()(conv3)
        
        # Concatenate pooled features
        concat = Concatenate()([pool1, pool2, pool3])
        text_vector = Dense(64, activation='relu')(concat)
        
        return text_vector
    
    def _build_model(self):
        """
        Build the complete multi-modal recommendation model
        
        Returns:
            Compiled Keras model
        """
        # Input layers
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')
        image_input = Input(shape=self.image_input_shape, name='image_input')
        text_input = Input(shape=(self.text_input_length,), name='text_input')
        
        # Build individual branches
        cf_vector = self._build_collaborative_filtering_branch(user_input, item_input)
        image_vector = self._build_image_branch(image_input)
        text_vector = self._build_text_branch(text_input)
        
        # Fusion of modalities
        fusion = Concatenate()([cf_vector, image_vector, text_vector])
        fusion = Dense(64, activation='relu')(fusion)
        fusion = Dropout(0.3)(fusion)
        fusion = Dense(32, activation='relu')(fusion)
        
        # Output layer
        output = Dense(1, activation='sigmoid', name='prediction')(fusion)
        
        # Create and compile the model
        model = Model(inputs=[user_input, item_input, image_input, text_input], 
                     outputs=output)
        
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        
        return model
    
    def fit_tokenizer(self, text_data):
        """
        Fit the text tokenizer on the training data
        
        Args:
            text_data: List of text descriptions/reviews
        """
        self.tokenizer = Tokenizer(num_words=self.text_vocab_size)
        self.tokenizer.fit_on_texts(text_data)
        self.logger.info(f"Tokenizer fitted with vocabulary size: {len(self.tokenizer.word_index)}")
    
    def preprocess_text(self, text_data):
        """
        Preprocess text data using the fitted tokenizer
        
        Args:
            text_data: List of text descriptions/reviews
            
        Returns:
            Padded sequences of tokenized text
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been fitted yet. Call fit_tokenizer first.")
            
        sequences = self.tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(sequences, maxlen=self.text_input_length)
        return padded_sequences
    
    def fit(self, user_ids, item_ids, images, texts, ratings, 
            epochs=10, batch_size=64, validation_split=0.1, verbose=1):
        """
        Train the multi-modal recommendation model
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            images: Array of image data (n_samples, height, width, channels)
            texts: Array of preprocessed text sequences
            ratings: Array of ratings (target values)
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of data to use for validation
            verbose: Verbosity mode
            
        Returns:
            Training history
        """
        self.logger.info(f"Training multi-modal model with {len(user_ids)} samples")
        
        history = self.model.fit(
            [np.array(user_ids), np.array(item_ids), np.array(images), np.array(texts)],
            np.array(ratings),
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        
        return history
    
    def predict(self, user_ids, item_ids, images, texts):
        """
        Generate predictions for the given user-item pairs with content
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            images: Array of image data
            texts: Array of preprocessed text sequences
            
        Returns:
            Predicted ratings
        """
        predictions = self.model.predict(
            [np.array(user_ids), np.array(item_ids), np.array(images), np.array(texts)])
        return predictions
    
    def recommend(self, user_id, item_data, n_recommendations=5, exclude_rated=True, rated_items=None):
        """
        Generate recommendations for a user
        
        Args:
            user_id: User ID to recommend for
            item_data: Dictionary with keys 'item_ids', 'images', 'texts' containing data for all items
            n_recommendations: Number of recommendations to generate
            exclude_rated: Whether to exclude already rated items
            rated_items: List of items already rated by the user (required if exclude_rated=True)
            
        Returns:
            List of (item_id, predicted_rating) tuples
        """
        if exclude_rated and rated_items is None:
            raise ValueError("rated_items must be provided if exclude_rated=True")
        
        # Filter out already rated items if needed
        if exclude_rated:
            mask = np.isin(item_data['item_ids'], rated_items, invert=True)
            item_ids = item_data['item_ids'][mask]
            images = item_data['images'][mask]
            texts = item_data['texts'][mask]
        else:
            item_ids = item_data['item_ids']
            images = item_data['images']
            texts = item_data['texts']
        
        # Create user-item pairs for prediction
        user_ids = np.full(len(item_ids), user_id)
        
        # Get predictions
        predictions = self.predict(user_ids, item_ids, images, texts).flatten()
        
        # Create (item_id, rating) pairs and sort by rating
        item_rating_pairs = list(zip(item_ids, predictions))
        item_rating_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return item_rating_pairs[:n_recommendations]
    
    def save_model(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load the model from a file
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from {filepath}")


def prepare_multi_modal_data(ratings_df, item_metadata_df, image_path, text_column='description'):
    """
    Prepare data for the multi-modal recommender
    
    Args:
        ratings_df: DataFrame with user-item ratings
        item_metadata_df: DataFrame with item metadata including text descriptions
        image_path: Path to the directory containing item images
        text_column: Column name in item_metadata_df containing text data
        
    Returns:
        Processed data ready for the multi-modal model
    """
    from sklearn.preprocessing import LabelEncoder
    import os
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    
    # Encode user and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    user_ids = user_encoder.fit_transform(ratings_df['user_id'].values)
    item_ids = item_encoder.fit_transform(ratings_df['movie_id'].values)
    
    # Normalize ratings to [0, 1]
    ratings = ratings_df['rating'].values / 5.0
    
    # Process images
    image_size = (224, 224)
    images = []
    
    for item_id in ratings_df['movie_id'].values:
        img_file = os.path.join(image_path, f"{item_id}.jpg")
        try:
            if os.path.exists(img_file):
                img = load_img(img_file, target_size=image_size)
                img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
            else:
                # Use a placeholder image if the actual image is not available
                img_array = np.zeros((*image_size, 3))
        except Exception as e:
            print(f"Error loading image for item {item_id}: {e}")
            img_array = np.zeros((*image_size, 3))
        
        images.append(img_array)
    
    # Process text data
    texts = []
    
    for item_id in ratings_df['movie_id'].values:
        if item_id in item_metadata_df.index:
            text = item_metadata_df.loc[item_id, text_column]
        else:
            text = ""  # Use empty string if no description is available
        
        texts.append(text)
    
    # Create a multi-modal recommender instance
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    recommender = MultiModalRecommender(
        n_users=n_users,
        n_items=n_items
    )
    
    # Fit the tokenizer and preprocess text
    recommender.fit_tokenizer(texts)
    processed_texts = recommender.preprocess_text(texts)
    
    return {
        'user_ids': user_ids,
        'item_ids': item_ids,
        'images': np.array(images),
        'texts': processed_texts,
        'ratings': ratings,
        'n_users': n_users,
        'n_items': n_items,
        'user_encoder': user_encoder,
        'item_encoder': item_encoder,
        'recommender': recommender
    }


def train_multi_modal_model(data, epochs=10, batch_size=64):
    """
    Train the multi-modal recommendation model
    
    Args:
        data: Dictionary with processed data from prepare_multi_modal_data
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        Trained model and training history
    """
    recommender = data['recommender']
    
    history = recommender.fit(
        user_ids=data['user_ids'],
        item_ids=data['item_ids'],
        images=data['images'],
        texts=data['texts'],
        ratings=data['ratings'],
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    
    return recommender, history


def get_multi_modal_recommendations(model, user_id, item_data, user_encoder, item_encoder, 
                                  n_recommendations=5, exclude_rated=True, ratings_df=None):
    """
    Get recommendations for a user using the multi-modal model
    
    Args:
        model: Trained multi-modal model
        user_id: Original user ID (before encoding)
        item_data: Dictionary with item data (item_ids, images, texts)
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
    
    # Get rated items if needed
    rated_items = None
    if exclude_rated and ratings_df is not None:
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        rated_items = item_encoder.transform(user_ratings['movie_id'].values)
    
    # Get recommendations
    recommendations = model.recommend(
        user_id=encoded_user_id,
        item_data=item_data,
        n_recommendations=n_recommendations,
        exclude_rated=exclude_rated,
        rated_items=rated_items
    )
    
    # Decode item IDs
    decoded_recommendations = [
        (item_encoder.inverse_transform([int(item_id)])[0], rating)
        for item_id, rating in recommendations
    ]
    
    return decoded_recommendations


class MultiModalRecommenderWrapper:
    """
    Wrapper class for the MultiModalRecommender to provide a consistent interface
    for the demo application
    """
    def __init__(self):
        self.recommender = None
        self.user_encoder = None
        self.item_encoder = None
        self.data = None
        self.logger = logging.getLogger(__name__)
    
    def train(self, users, items, interactions):
        """
        Train the multi-modal recommender model
        
        Args:
            users: DataFrame with user information
            items: DataFrame with item information
            interactions: DataFrame with user-item interactions
        """
        # Prepare data for training
        # For demo purposes, we'll create mock image and text data
        self.logger.info("Preparing data for multi-modal recommender training")
        
        # Use a smaller sample size for demo purposes to avoid memory issues
        if len(interactions) > 50:
            interactions = interactions.sample(50, random_state=42)
        
        # Generate memory-efficient image data in batches
        image_size = (16, 16)
        # Memory-efficient image processing
        batch_size = 16
        mock_images = np.zeros((len(interactions), *image_size, 3))
        for i in range(0, len(interactions), batch_size):
            batch = interactions[i:i+batch_size]
        mock_images[i:i+batch_size] = np.zeros((len(batch), *image_size, 3))
        
        # Create mock text data
        mock_texts = [f"Description for item {item_id}" for item_id in interactions['item_id']]
        
        # Prepare the data
        from sklearn.preprocessing import LabelEncoder
        
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        user_ids = self.user_encoder.fit_transform(interactions['user_id'].values)
        item_ids = self.item_encoder.fit_transform(interactions['item_id'].values)
        
        # Normalize ratings to [0, 1]
        ratings = interactions['rating'].values / 5.0
        
        # Create and initialize the recommender
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        self.recommender = MultiModalRecommender(
            n_users=n_users,
            n_items=n_items,
            image_input_shape=(16, 16, 3)  # Match mock image dimensions
        )
        
        # Fit the tokenizer and preprocess text
        self.recommender.fit_tokenizer(mock_texts)
        processed_texts = self.recommender.preprocess_text(mock_texts)
        
        # Store the data for later use
        self.data = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'images': mock_images,
            'texts': processed_texts,
            'ratings': ratings
        }
        
        # Train the model
        self.logger.info("Training multi-modal recommender model")
        self.recommender.fit(
            user_ids=self.data['user_ids'],
            item_ids=self.data['item_ids'],
            images=self.data['images'],
            texts=self.data['texts'],
            ratings=self.data['ratings'],
            epochs=2,  # Reduced for demo purposes
            batch_size=16,  # Smaller batch size
            validation_split=0.1
        )
        
        self.logger.info("Multi-modal recommender training completed")
    
    def get_recommendations(self, user_id, users, items, top_n=5):
        """
        Get recommendations for a user
        
        Args:
            user_id: User ID to get recommendations for
            users: DataFrame with user information
            items: DataFrame with item information
            top_n: Number of recommendations to return
            
        Returns:
            DataFrame with recommendations
        """
        if self.recommender is None:
            raise ValueError("Recommender has not been trained yet")
        
        # Create item data for prediction
        # Limit the number of items to predict for to avoid memory issues
        if len(items) > 50:
            items_sample = items.sample(50, random_state=42)
            unique_items = items_sample['item_id'].unique()
        else:
            unique_items = items['item_id'].unique()
        
        # Filter items to only include those that were in the training data
        known_items = [item for item in unique_items if item in self.item_encoder.classes_]
        
        if not known_items:
            # If no known items, return empty recommendations
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])
            
        encoded_items = self.item_encoder.transform(known_items)
        
        # Create mock image and text data for all items with smaller dimensions
        image_size = (16, 16)  # Match model input dimensions
        # Generate prediction images in smaller batches
        batch_size = 16
        all_images = np.zeros((len(known_items), *image_size, 3))
        for i in range(0, len(known_items), batch_size):
            batch_items = known_items[i:i+batch_size]
            all_images[i:i+batch_size] = np.random.random((len(batch_items), *image_size, 3))
        all_texts = [f"Description for item {item_id}" for item_id in known_items]
        processed_texts = self.recommender.preprocess_text(all_texts)
        
        item_data = {
            'item_ids': encoded_items,
            'images': all_images,
            'texts': processed_texts
        }
        
        # Get recommendations
        try:
            encoded_user_id = self.user_encoder.transform([user_id])[0]
        except ValueError:
            # If user is not in the training data, return empty recommendations
            return pd.DataFrame(columns=['item_id', 'predicted_rating'])
        
        recommendations = self.recommender.recommend(
            user_id=encoded_user_id,
            item_data=item_data,
            n_recommendations=min(top_n, len(unique_items)),
            exclude_rated=False  # For demo purposes
        )
        
        # Convert to DataFrame
        rec_items = [self.item_encoder.inverse_transform([int(item_id)])[0] for item_id, _ in recommendations]
        rec_ratings = [float(rating) * 5.0 for _, rating in recommendations]  # Scale back to 1-5
        
        recommendations_df = pd.DataFrame({
            'item_id': rec_items,
            'predicted_rating': rec_ratings
        })
        
        return recommendations_df


def create_multi_modal_recommender():
    """
    Create and return a multi-modal recommender instance
    
    Returns:
        MultiModalRecommenderWrapper instance
    """
    return MultiModalRecommenderWrapper()
    # Add temporal attention layer
    x = Attention()([x, timestamps])