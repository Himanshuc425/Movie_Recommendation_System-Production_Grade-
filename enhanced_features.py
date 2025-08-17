import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class TemporalFeatureExtractor:
    """
    Extract temporal features from timestamp data
    
    This class generates time-based features that capture patterns in user-item interactions
    over time, such as seasonality, recency, and frequency.
    """
    def __init__(self, timestamp_col='timestamp'):
        """
        Initialize the temporal feature extractor
        
        Args:
            timestamp_col: Name of the timestamp column
        """
        self.timestamp_col = timestamp_col
        
    def fit(self, X, y=None):
        """
        Fit the feature extractor (no-op for this extractor)
        
        Args:
            X: DataFrame containing the timestamp column
            y: Target variable (not used)
            
        Returns:
            self
        """
        return self
    
    def transform(self, X):
        """
        Transform the data by extracting temporal features
        
        Args:
            X: DataFrame containing the timestamp column
            
        Returns:
            DataFrame with extracted temporal features
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Convert timestamp to datetime if it's not already
        if X_transformed[self.timestamp_col].dtype != 'datetime64[ns]':
            X_transformed[self.timestamp_col] = pd.to_datetime(X_transformed[self.timestamp_col], unit='s')
        
        # Extract basic time components
        X_transformed['hour_of_day'] = X_transformed[self.timestamp_col].dt.hour
        X_transformed['day_of_week'] = X_transformed[self.timestamp_col].dt.dayofweek
        X_transformed['day_of_month'] = X_transformed[self.timestamp_col].dt.day
        X_transformed['month_of_year'] = X_transformed[self.timestamp_col].dt.month
        X_transformed['year'] = X_transformed[self.timestamp_col].dt.year
        X_transformed['is_weekend'] = X_transformed['day_of_week'].isin([5, 6]).astype(int)
        
        # Create cyclical features for time components
        # This preserves the cyclical nature of time features (e.g., hour 23 is close to hour 0)
        X_transformed['hour_sin'] = np.sin(2 * np.pi * X_transformed['hour_of_day'] / 24)
        X_transformed['hour_cos'] = np.cos(2 * np.pi * X_transformed['hour_of_day'] / 24)
        X_transformed['day_sin'] = np.sin(2 * np.pi * X_transformed['day_of_week'] / 7)
        X_transformed['day_cos'] = np.cos(2 * np.pi * X_transformed['day_of_week'] / 7)
        X_transformed['month_sin'] = np.sin(2 * np.pi * X_transformed['month_of_year'] / 12)
        X_transformed['month_cos'] = np.cos(2 * np.pi * X_transformed['month_of_year'] / 12)
        
        # Calculate recency features
        max_timestamp = X_transformed[self.timestamp_col].max()
        X_transformed['days_since_last_interaction'] = (max_timestamp - X_transformed[self.timestamp_col]).dt.total_seconds() / (24 * 3600)
        
        # Drop the original timestamp column
        X_transformed = X_transformed.drop(self.timestamp_col, axis=1)
        
        return X_transformed


class UserContextFeatureExtractor:
    """
    Extract contextual features related to user behavior
    
    This class generates features that capture the context of user interactions,
    such as user activity patterns, preferences, and behavior over time.
    """
    def __init__(self, user_col='user_id', item_col='movie_id', rating_col='rating', timestamp_col=None):
        """
        Initialize the user context feature extractor
        
        Args:
            user_col: Name of the user column
            item_col: Name of the item column
            rating_col: Name of the rating column
            timestamp_col: Name of the timestamp column (optional)
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.user_stats = None
        
    def fit(self, X, y=None):
        """
        Fit the feature extractor by calculating user statistics
        
        Args:
            X: DataFrame containing user-item interactions
            y: Target variable (not used)
            
        Returns:
            self
        """
        # Calculate user statistics
        self.user_stats = X.groupby(self.user_col).agg({
            self.rating_col: ['count', 'mean', 'std', 'min', 'max'],
            self.item_col: 'nunique'
        })
        
        # Flatten the column names
        self.user_stats.columns = ['_'.join(col).strip() for col in self.user_stats.columns.values]
        
        # Rename columns for clarity
        self.user_stats = self.user_stats.rename(columns={
            f'{self.rating_col}_count': 'user_total_ratings',
            f'{self.rating_col}_mean': 'user_mean_rating',
            f'{self.rating_col}_std': 'user_rating_std',
            f'{self.rating_col}_min': 'user_min_rating',
            f'{self.rating_col}_max': 'user_max_rating',
            f'{self.item_col}_nunique': 'user_unique_items'
        })
        
        # Fill NaN values in standard deviation with 0 (for users with only one rating)
        self.user_stats['user_rating_std'] = self.user_stats['user_rating_std'].fillna(0)
        
        # Calculate additional user features
        if self.user_stats['user_total_ratings'].max() > 0:
            self.user_stats['user_activity_level'] = self.user_stats['user_total_ratings'] / self.user_stats['user_total_ratings'].max()
        else:
            self.user_stats['user_activity_level'] = 0
            
        self.user_stats['user_rating_range'] = self.user_stats['user_max_rating'] - self.user_stats['user_min_rating']
        
        return self
    
    def transform(self, X):
        """
        Transform the data by adding user context features
        
        Args:
            X: DataFrame containing user-item interactions
            
        Returns:
            DataFrame with added user context features
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Merge user statistics with the original data
        X_transformed = X_transformed.merge(
            self.user_stats.reset_index(),
            on=self.user_col,
            how='left'
        )
        
        # Fill NaN values for new users
        user_stats_cols = self.user_stats.columns.tolist()
        X_transformed[user_stats_cols] = X_transformed[user_stats_cols].fillna({
            'user_total_ratings': 0,
            'user_mean_rating': X_transformed[self.rating_col].mean(),
            'user_rating_std': 0,
            'user_min_rating': X_transformed[self.rating_col].min(),
            'user_max_rating': X_transformed[self.rating_col].max(),
            'user_unique_items': 0,
            'user_activity_level': 0,
            'user_rating_range': 0
        })
        
        return X_transformed


class ItemContextFeatureExtractor:
    """
    Extract contextual features related to items
    
    This class generates features that capture the context of items,
    such as popularity, average ratings, and genre information.
    """
    def __init__(self, user_col='user_id', item_col='movie_id', rating_col='rating', genre_col=None):
        """
        Initialize the item context feature extractor
        
        Args:
            user_col: Name of the user column
            item_col: Name of the item column
            rating_col: Name of the rating column
            genre_col: Name of the genre column (optional)
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.genre_col = genre_col
        self.item_stats = None
        self.genre_encoder = None
        
    def fit(self, X, y=None):
        """
        Fit the feature extractor by calculating item statistics
        
        Args:
            X: DataFrame containing user-item interactions
            y: Target variable (not used)
            
        Returns:
            self
        """
        # Calculate item statistics
        self.item_stats = X.groupby(self.item_col).agg({
            self.rating_col: ['count', 'mean', 'std', 'min', 'max'],
            self.user_col: 'nunique'
        })
        
        # Flatten the column names
        self.item_stats.columns = ['_'.join(col).strip() for col in self.item_stats.columns.values]
        
        # Rename columns for clarity
        self.item_stats = self.item_stats.rename(columns={
            f'{self.rating_col}_count': 'item_total_ratings',
            f'{self.rating_col}_mean': 'item_mean_rating',
            f'{self.rating_col}_std': 'item_rating_std',
            f'{self.rating_col}_min': 'item_min_rating',
            f'{self.rating_col}_max': 'item_max_rating',
            f'{self.user_col}_nunique': 'item_unique_users'
        })
        
        # Fill NaN values in standard deviation with 0 (for items with only one rating)
        self.item_stats['item_rating_std'] = self.item_stats['item_rating_std'].fillna(0)
        
        # Calculate additional item features
        if self.item_stats['item_total_ratings'].max() > 0:
            self.item_stats['item_popularity'] = self.item_stats['item_total_ratings'] / self.item_stats['item_total_ratings'].max()
        else:
            self.item_stats['item_popularity'] = 0
            
        self.item_stats['item_rating_range'] = self.item_stats['item_max_rating'] - self.item_stats['item_min_rating']
        
        # Process genre information if available
        if self.genre_col and self.genre_col in X.columns:
            # Create a one-hot encoder for genres
            # First, explode the genre column if it contains lists or comma-separated values
            if X[self.genre_col].dtype == 'object':
                # Check if the genre column contains comma-separated values
                if X[self.genre_col].str.contains(',').any():
                    # Split and explode the genre column
                    genres_exploded = X[[self.item_col, self.genre_col]].copy()
                    genres_exploded[self.genre_col] = genres_exploded[self.genre_col].str.split(',').apply(lambda x: [g.strip() for g in x])
                    genres_exploded = genres_exploded.explode(self.genre_col)
                else:
                    genres_exploded = X[[self.item_col, self.genre_col]]
                    
                # Create a one-hot encoder for genres
                self.genre_encoder = OneHotEncoder(sparse_output=False)
                self.genre_encoder.fit(genres_exploded[[self.genre_col]])
                
                # Get the genre features for each item
                genre_features = pd.DataFrame(
                    self.genre_encoder.transform(genres_exploded[[self.genre_col]]),
                    columns=self.genre_encoder.get_feature_names_out([self.genre_col])
                )
                
                # Add the item_id back to the genre features
                genre_features[self.item_col] = genres_exploded[self.item_col].values
                
                # Aggregate genre features by item (using max to ensure binary values)
                genre_features = genre_features.groupby(self.item_col).max()
                
                # Merge genre features with item statistics
                self.item_stats = self.item_stats.join(genre_features)
        
        return self
    
    def transform(self, X):
        """
        Transform the data by adding item context features
        
        Args:
            X: DataFrame containing user-item interactions
            
        Returns:
            DataFrame with added item context features
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Merge item statistics with the original data
        X_transformed = X_transformed.merge(
            self.item_stats.reset_index(),
            on=self.item_col,
            how='left'
        )
        
        # Fill NaN values for new items
        item_stats_cols = [col for col in self.item_stats.columns if not col.startswith(self.genre_col + '_')]
        X_transformed[item_stats_cols] = X_transformed[item_stats_cols].fillna({
            'item_total_ratings': 0,
            'item_mean_rating': X_transformed[self.rating_col].mean(),
            'item_rating_std': 0,
            'item_min_rating': X_transformed[self.rating_col].min(),
            'item_max_rating': X_transformed[self.rating_col].max(),
            'item_unique_users': 0,
            'item_popularity': 0,
            'item_rating_range': 0
        })
        
        # Fill NaN values for genre features (if they exist)
        genre_cols = [col for col in self.item_stats.columns if col.startswith(self.genre_col + '_')]
        if genre_cols:
            X_transformed[genre_cols] = X_transformed[genre_cols].fillna(0)
        
        return X_transformed


class InteractionContextFeatureExtractor:
    """
    Extract contextual features from user-item interactions
    
    This class generates features that capture the context of specific user-item interactions,
    such as user-item affinity, cross-features, and sequential patterns.
    """
    def __init__(self, user_col='user_id', item_col='movie_id', rating_col='rating', timestamp_col=None):
        """
        Initialize the interaction context feature extractor
        
        Args:
            user_col: Name of the user column
            item_col: Name of the item column
            rating_col: Name of the rating column
            timestamp_col: Name of the timestamp column (optional)
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.global_mean = None
        self.user_means = None
        self.item_means = None
        
    def fit(self, X, y=None):
        """
        Fit the feature extractor by calculating global, user, and item means
        
        Args:
            X: DataFrame containing user-item interactions
            y: Target variable (not used)
            
        Returns:
            self
        """
        # Calculate global mean rating
        self.global_mean = X[self.rating_col].mean()
        
        # Calculate user mean ratings
        self.user_means = X.groupby(self.user_col)[self.rating_col].mean()
        
        # Calculate item mean ratings
        self.item_means = X.groupby(self.item_col)[self.rating_col].mean()
        
        return self
    
    def transform(self, X):
        """
        Transform the data by adding interaction context features
        
        Args:
            X: DataFrame containing user-item interactions
            
        Returns:
            DataFrame with added interaction context features
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Add user mean rating
        X_transformed['user_mean_rating'] = X_transformed[self.user_col].map(self.user_means)
        X_transformed['user_mean_rating'] = X_transformed['user_mean_rating'].fillna(self.global_mean)
        
        # Add item mean rating
        X_transformed['item_mean_rating'] = X_transformed[self.item_col].map(self.item_means)
        X_transformed['item_mean_rating'] = X_transformed['item_mean_rating'].fillna(self.global_mean)
        
        # Calculate user-item affinity (deviation from expected rating)
        X_transformed['user_item_affinity'] = X_transformed[self.rating_col] - (
            X_transformed['user_mean_rating'] + X_transformed['item_mean_rating'] - self.global_mean
        )
        
        # Create cross-features
        X_transformed['user_item_interaction'] = X_transformed[self.user_col].astype(str) + '_' + X_transformed[self.item_col].astype(str)
        
        # Add sequential features if timestamp is available
        if self.timestamp_col and self.timestamp_col in X_transformed.columns:
            # Convert timestamp to datetime if it's not already
            if X_transformed[self.timestamp_col].dtype != 'datetime64[ns]':
                X_transformed[self.timestamp_col] = pd.to_datetime(X_transformed[self.timestamp_col], unit='s')
            
            # Sort by user and timestamp
            X_transformed = X_transformed.sort_values([self.user_col, self.timestamp_col])
            
            # Calculate time since last rating for each user
            X_transformed['prev_timestamp'] = X_transformed.groupby(self.user_col)[self.timestamp_col].shift(1)
            X_transformed['time_since_last_rating'] = (X_transformed[self.timestamp_col] - X_transformed['prev_timestamp']).dt.total_seconds() / 3600  # in hours
            X_transformed['time_since_last_rating'] = X_transformed['time_since_last_rating'].fillna(0)
            
            # Calculate previous rating features
            X_transformed['prev_rating'] = X_transformed.groupby(self.user_col)[self.rating_col].shift(1)
            X_transformed['prev_rating'] = X_transformed['prev_rating'].fillna(X_transformed['user_mean_rating'])
            
            # Calculate rating change
            X_transformed['rating_change'] = X_transformed[self.rating_col] - X_transformed['prev_rating']
            
            # Drop temporary columns
            X_transformed = X_transformed.drop('prev_timestamp', axis=1)
        
        return X_transformed


class EnhancedFeatureEngineering:
    """
    Comprehensive feature engineering pipeline for recommendation systems
    
    This class combines temporal, user context, item context, and interaction context
    features to create a rich feature set for recommendation models.
    """
    def __init__(self, user_col='user_id', item_col='movie_id', rating_col='rating', 
                 timestamp_col=None, genre_col=None, text_cols=None):
        """
        Initialize the enhanced feature engineering pipeline
        
        Args:
            user_col: Name of the user column
            item_col: Name of the item column
            rating_col: Name of the rating column
            timestamp_col: Name of the timestamp column (optional)
            genre_col: Name of the genre column (optional)
            text_cols: List of text columns to process (optional)
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.genre_col = genre_col
        self.text_cols = text_cols or []
        
        # Initialize feature extractors
        self.temporal_extractor = None
        if timestamp_col:
            self.temporal_extractor = TemporalFeatureExtractor(timestamp_col=timestamp_col)
            
        self.user_context_extractor = UserContextFeatureExtractor(
            user_col=user_col, item_col=item_col, rating_col=rating_col, timestamp_col=timestamp_col
        )
        
        self.item_context_extractor = ItemContextFeatureExtractor(
            user_col=user_col, item_col=item_col, rating_col=rating_col, genre_col=genre_col
        )
        
        self.interaction_extractor = InteractionContextFeatureExtractor(
            user_col=user_col, item_col=item_col, rating_col=rating_col, timestamp_col=timestamp_col
        )
        
        # Text processing pipeline
        self.text_processors = {}
        for col in self.text_cols:
            self.text_processors[col] = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=100, stop_words='english')),
                ('svd', TruncatedSVD(n_components=10))
            ])
        
        # Numerical feature scaler
        self.scaler = StandardScaler()
        self.numerical_cols = []
        
    def fit(self, X, y=None):
        """Fit the feature engineering pipeline
        
        Args:
            X: DataFrame containing user-item interactions
            y: Target variable (not used)
            
        Returns:
            self
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Apply each feature extractor
        if self.temporal_extractor:
            X_transformed = self.temporal_extractor.fit(X_transformed).transform(X_transformed)
            
        X_transformed = self.user_context_extractor.fit(X_transformed).transform(X_transformed)
        X_transformed = self.item_context_extractor.fit(X_transformed).transform(X_transformed)
        X_transformed = self.interaction_extractor.fit(X_transformed).transform(X_transformed)
        
        # Process text columns
        for col in self.text_cols:
            if col in X_transformed.columns:
                # Fill NaN values
                X_transformed[col] = X_transformed[col].fillna('')
                # Fit the text processor
                self.text_processors[col].fit(X_transformed[col])
        
        # Identify numerical columns for scaling
        self.numerical_cols = X_transformed.select_dtypes(include=['float64', 'int64']).columns.tolist()
        # Remove the target column from scaling
        if self.rating_col in self.numerical_cols:
            self.numerical_cols.remove(self.rating_col)
        # Also remove user and item IDs from scaling
        for col in [self.user_col, self.item_col]:
            if col in self.numerical_cols:
                self.numerical_cols.remove(col)
        
        # Fit the scaler on numerical columns
        if self.numerical_cols:
            self.scaler.fit(X_transformed[self.numerical_cols])
        
        return self
    
    def transform(self, X):
        """
        Transform the data by applying all feature extractors
        
        Args:
            X: DataFrame containing user-item interactions
            
        Returns:
            DataFrame with all engineered features
        """
        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()
        
        # Apply each feature extractor
        if self.temporal_extractor:
            X_transformed = self.temporal_extractor.transform(X_transformed)
            
        X_transformed = self.user_context_extractor.transform(X_transformed)
        X_transformed = self.item_context_extractor.transform(X_transformed)
        X_transformed = self.interaction_extractor.transform(X_transformed)
        
        # Process text columns
        for col in self.text_cols:
            if col in X_transformed.columns:
                # Fill NaN values
                X_transformed[col] = X_transformed[col].fillna('')
                # Transform the text data
                text_features = self.text_processors[col].transform(X_transformed[col])
                # Convert to DataFrame
                text_features_df = pd.DataFrame(
                    text_features,
                    columns=[f'{col}_svd_{i}' for i in range(text_features.shape[1])]
                )
                # Add to the transformed data
                X_transformed = pd.concat([X_transformed, text_features_df], axis=1)
                # Drop the original text column
                X_transformed = X_transformed.drop(col, axis=1)
        
        # Scale numerical columns
        if self.numerical_cols:
            # Only scale columns that exist in the data
            cols_to_scale = [col for col in self.numerical_cols if col in X_transformed.columns]
            if cols_to_scale:
                X_transformed[cols_to_scale] = self.scaler.transform(X_transformed[cols_to_scale])
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform the data
        
        Args:
            X: DataFrame containing user-item interactions
            y: Target variable (not used)
            
        Returns:
            DataFrame with all engineered features
        """
        return self.fit(X, y).transform(X)


def add_synthetic_temporal_data(ratings_df, start_date='2020-01-01', end_date='2023-01-01'):
    """
    Add synthetic timestamp data to a ratings DataFrame
    
    Args:
        ratings_df: DataFrame with columns 'user_id', 'movie_id', 'rating'
        start_date: Start date for synthetic timestamps
        end_date: End date for synthetic timestamps
        
    Returns:
        DataFrame with added 'timestamp' column
    """
    # Convert dates to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Calculate the total number of seconds in the date range
    total_seconds = (end_date - start_date).total_seconds()
    
    # Generate random timestamps within the date range
    np.random.seed(42)  # For reproducibility
    random_seconds = np.random.randint(0, int(total_seconds), size=len(ratings_df))
    timestamps = [start_date + timedelta(seconds=int(sec)) for sec in random_seconds]
    
    # Add timestamps to the DataFrame
    ratings_df_with_time = ratings_df.copy()
    ratings_df_with_time['timestamp'] = timestamps
    
    # Sort by user_id and timestamp
    ratings_df_with_time = ratings_df_with_time.sort_values(['user_id', 'timestamp'])
    
    return ratings_df_with_time


def add_synthetic_context_data(ratings_df, movies_df=None):
    """
    Add synthetic contextual data to a ratings DataFrame
    
    Args:
        ratings_df: DataFrame with columns 'user_id', 'movie_id', 'rating'
        movies_df: DataFrame with movie metadata (optional)
        
    Returns:
        DataFrame with added contextual columns
    """
    # Make a copy of the ratings DataFrame
    enhanced_df = ratings_df.copy()
    
    # Add device information
    devices = ['mobile', 'tablet', 'desktop', 'tv', 'console']
    enhanced_df['device'] = np.random.choice(devices, size=len(enhanced_df))
    
    # Add location information
    locations = ['home', 'work', 'traveling', 'commuting', 'other']
    enhanced_df['location'] = np.random.choice(locations, size=len(enhanced_df))
    
    # Add mood information
    moods = ['happy', 'sad', 'neutral', 'excited', 'relaxed']
    enhanced_df['mood'] = np.random.choice(moods, size=len(enhanced_df))
    
    # Add social context
    social_contexts = ['alone', 'with_family', 'with_friends', 'with_partner', 'other']
    enhanced_df['social_context'] = np.random.choice(social_contexts, size=len(enhanced_df))
    
    # Add time spent watching
    # Generate random values between 0.5 and 1.5 times the movie duration (if available)
    if movies_df is not None and 'duration' in movies_df.columns:
        # Merge movie durations
        enhanced_df = enhanced_df.merge(movies_df[['movie_id', 'duration']], on='movie_id', how='left')
        # Fill missing durations with median
        median_duration = movies_df['duration'].median()
        enhanced_df['duration'] = enhanced_df['duration'].fillna(median_duration)
        # Generate time spent watching
        enhanced_df['time_spent'] = enhanced_df['duration'] * np.random.uniform(0.5, 1.5, size=len(enhanced_df))
    else:
        # Generate random time spent between 30 and 180 minutes
        enhanced_df['time_spent'] = np.random.uniform(30, 180, size=len(enhanced_df))
    
    return enhanced_df