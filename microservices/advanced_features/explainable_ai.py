import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
import shap
import lime
import lime.lime_tabular
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import logging


def create_explainable_recommender(base_model=None, user_features=None, item_features=None):
    """
    Create and return an explainable recommender instance
    
    Args:
        base_model: The underlying recommendation model to explain
        user_features: DataFrame or array of user features
        item_features: DataFrame or array of item features
        
    Returns:
        ExplainableRecommender instance
    """
    return ExplainableRecommender(base_model, user_features, item_features)


def get_explanation(recommender, user_id, item_id, explanation_type='feature_importance'):
    """
    Get explanation for a recommendation
    
    Args:
        recommender: ExplainableRecommender instance
        user_id: User ID to explain recommendation for
        item_id: Item ID to explain recommendation for
        explanation_type: Type of explanation to generate
        
    Returns:
        Dictionary with explanation data
    """
    if explanation_type == 'feature_importance':
        return recommender.explain_feature_importance(user_id, item_id)
    elif explanation_type == 'similar_items':
        return recommender.explain_similar_items(item_id)
    elif explanation_type == 'user_preferences':
        return recommender.explain_user_preferences(user_id)
    else:
        raise ValueError(f"Unsupported explanation type: {explanation_type}")


class ExplainableRecommender:
    """
    Explainable Recommender System that provides transparency and interpretability
    for recommendation results using various explainability techniques.
    
    This class wraps around existing recommendation models and adds explainability
    features to help users understand why certain items are being recommended.
    """
    
    def __init__(self, base_model, user_features=None, item_features=None, 
                 user_item_matrix=None, item_metadata=None):
        """
        Initialize the explainable recommender.
        
        Args:
            base_model: The underlying recommendation model to explain
            user_features: DataFrame or array of user features
            item_features: DataFrame or array of item features
            user_item_matrix: Matrix of user-item interactions
            item_metadata: DataFrame with item metadata for content-based explanations
        """
        self.base_model = base_model
        self.user_features = user_features
        self.item_features = item_features
        self.user_item_matrix = user_item_matrix
        self.item_metadata = item_metadata
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        
        # For content-based explanations
        self.item_similarity_matrix = None
        if item_features is not None:
            self._compute_item_similarity()
        
        # Logger setup
        self.logger = logging.getLogger(__name__)
    
    def _compute_item_similarity(self):
        """
        Compute item-item similarity matrix based on item features.
        """
        if isinstance(self.item_features, pd.DataFrame):
            features = self.item_features.values
        else:
            features = self.item_features
            
        self.item_similarity_matrix = cosine_similarity(features)
        self.logger.info(f"Computed item similarity matrix with shape {self.item_similarity_matrix.shape}")
    
    def initialize_shap_explainer(self, background_data=None):
        """
        Initialize the SHAP explainer for the base model.
        
        Args:
            background_data: Background data for the SHAP explainer
        """
        try:
            # Check if the model is a Keras model
            if isinstance(self.base_model, tf.keras.Model):
                if background_data is None:
                    self.logger.warning("No background data provided for SHAP explainer. Using random samples.")
                    # Create random background data if none provided
                    input_shape = self.base_model.input_shape
                    if isinstance(input_shape, list):
                        background_data = [np.random.random((100, shape[1])) for shape in input_shape]
                    else:
                        background_data = np.random.random((100, input_shape[1]))
                
                self.shap_explainer = shap.DeepExplainer(self.base_model, background_data)
                self.logger.info("SHAP explainer initialized for deep learning model")
            else:
                # For other model types (e.g., scikit-learn models)
                self.shap_explainer = shap.KernelExplainer(self.base_model.predict, background_data)
                self.logger.info("SHAP explainer initialized for ML model")
        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
    
    def initialize_lime_explainer(self, feature_names=None, class_names=None, categorical_features=None):
        """
        Initialize the LIME explainer for the base model.
        
        Args:
            feature_names: List of feature names
            class_names: List of class names
            categorical_features: List of indices of categorical features
        """
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.user_item_matrix if self.user_item_matrix is not None else np.random.random((100, 10)),
                feature_names=feature_names,
                class_names=class_names,
                categorical_features=categorical_features,
                mode='regression'
            )
            self.logger.info("LIME explainer initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize LIME explainer: {str(e)}")
    
    def explain_with_shap(self, input_data, plot=False):
        """
        Generate SHAP explanations for a prediction.
        
        Args:
            input_data: Input data for which to generate explanations
            plot: Whether to generate and return a plot
            
        Returns:
            SHAP values and optionally a plot
        """
        if self.shap_explainer is None:
            self.logger.warning("SHAP explainer not initialized. Initializing with input data.")
            self.initialize_shap_explainer(input_data)
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer.shap_values(input_data)
            
            # Generate plot if requested
            if plot:
                plt.figure(figsize=(10, 6))
                if isinstance(input_data, list):
                    # For multi-input models
                    shap.summary_plot(shap_values[0], input_data[0], show=False)
                else:
                    shap.summary_plot(shap_values, input_data, show=False)
                plt.tight_layout()
                return shap_values, plt.gcf()
            
            return shap_values
        except Exception as e:
            self.logger.error(f"Error generating SHAP explanations: {str(e)}")
            return None
    
    def explain_with_lime(self, input_instance, predict_fn=None, num_features=10):
        """
        Generate LIME explanations for a prediction.
        
        Args:
            input_instance: Input instance for which to generate explanations
            predict_fn: Prediction function (defaults to base_model.predict)
            num_features: Number of features to include in the explanation
            
        Returns:
            LIME explanation object
        """
        if self.lime_explainer is None:
            self.logger.warning("LIME explainer not initialized. Initializing with default settings.")
            self.initialize_lime_explainer()
        
        if predict_fn is None:
            predict_fn = self.base_model.predict
        
        try:
            explanation = self.lime_explainer.explain_instance(
                data_row=input_instance,
                predict_fn=predict_fn,
                num_features=num_features
            )
            return explanation
        except Exception as e:
            self.logger.error(f"Error generating LIME explanations: {str(e)}")
            return None
    
    def explain_by_similar_items(self, item_id, n_similar=5):
        """
        Explain a recommendation by showing similar items that the user liked.
        
        Args:
            item_id: ID of the item to explain
            n_similar: Number of similar items to include
            
        Returns:
            DataFrame of similar items with similarity scores
        """
        if self.item_similarity_matrix is None:
            if self.item_features is not None:
                self._compute_item_similarity()
            else:
                self.logger.error("Cannot explain by similar items: item features not provided")
                return None
        
        try:
            # Get similarity scores for the item
            item_idx = self._get_item_index(item_id)
            if item_idx is None:
                return None
                
            similarities = self.item_similarity_matrix[item_idx]
            
            # Get indices of most similar items (excluding the item itself)
            similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
            similar_scores = similarities[similar_indices]
            
            # Create a DataFrame with similar items
            similar_items = pd.DataFrame({
                'item_id': self._get_item_ids(similar_indices),
                'similarity': similar_scores
            })
            
            # Add item metadata if available
            if self.item_metadata is not None:
                similar_items = similar_items.merge(
                    self.item_metadata, on='item_id', how='left')
            
            return similar_items
        except Exception as e:
            self.logger.error(f"Error finding similar items: {str(e)}")
            return None
    
    def explain_by_user_history(self, user_id, item_id, n_items=5):
        """
        Explain a recommendation based on the user's rating history.
        
        Args:
            user_id: ID of the user
            item_id: ID of the recommended item
            n_items: Number of historical items to include
            
        Returns:
            DataFrame of relevant items from user history with explanation
        """
        if self.user_item_matrix is None:
            self.logger.error("Cannot explain by user history: user-item matrix not provided")
            return None
        
        try:
            # Get user and item indices
            user_idx = self._get_user_index(user_id)
            item_idx = self._get_item_index(item_id)
            
            if user_idx is None or item_idx is None:
                return None
            
            # Get items the user has rated
            if isinstance(self.user_item_matrix, pd.DataFrame):
                user_ratings = self.user_item_matrix.iloc[user_idx].to_numpy()
            else:
                user_ratings = self.user_item_matrix[user_idx]
            
            # Find rated items
            rated_indices = np.where(user_ratings > 0)[0]
            
            if len(rated_indices) == 0:
                self.logger.warning(f"User {user_id} has no ratings")
                return None
            
            # If we have item similarity, use it to find relevant items
            if self.item_similarity_matrix is not None:
                # Calculate relevance of each rated item to the recommended item
                relevance = self.item_similarity_matrix[item_idx, rated_indices]
                
                # Sort by relevance
                sorted_indices = np.argsort(relevance)[::-1][:n_items]
                relevant_item_indices = rated_indices[sorted_indices]
                relevance_scores = relevance[sorted_indices]
                
                # Get ratings for these items
                ratings = user_ratings[relevant_item_indices]
                
                # Create DataFrame
                history_items = pd.DataFrame({
                    'item_id': self._get_item_ids(relevant_item_indices),
                    'rating': ratings,
                    'relevance': relevance_scores
                })
            else:
                # Without similarity, just use highest rated items
                ratings = user_ratings[rated_indices]
                sorted_indices = np.argsort(ratings)[::-1][:n_items]
                relevant_item_indices = rated_indices[sorted_indices]
                
                # Create DataFrame
                history_items = pd.DataFrame({
                    'item_id': self._get_item_ids(relevant_item_indices),
                    'rating': ratings[sorted_indices]
                })
            
            # Add item metadata if available
            if self.item_metadata is not None:
                history_items = history_items.merge(
                    self.item_metadata, on='item_id', how='left')
            
            return history_items
        except Exception as e:
            self.logger.error(f"Error explaining by user history: {str(e)}")
            return None
    
    def explain_by_feature_contribution(self, user_id, item_id, top_n=5):
        """
        Explain a recommendation by showing the contribution of different features.
        
        Args:
            user_id: ID of the user
            item_id: ID of the recommended item
            top_n: Number of top features to include
            
        Returns:
            DataFrame of feature contributions
        """
        # This requires a model that can provide feature importances
        if not hasattr(self.base_model, 'feature_importances_') and self.shap_explainer is None:
            self.logger.warning("Model doesn't provide feature importances and SHAP explainer not initialized")
            return None
        
        try:
            # Get user and item indices
            user_idx = self._get_user_index(user_id)
            item_idx = self._get_item_index(item_id)
            
            if user_idx is None or item_idx is None:
                return None
            
            # Prepare input data for the model
            if hasattr(self, '_prepare_model_input'):
                input_data = self._prepare_model_input(user_idx, item_idx)
            else:
                # Default implementation - override in subclasses for specific models
                if self.user_features is not None and self.item_features is not None:
                    if isinstance(self.user_features, pd.DataFrame):
                        user_feat = self.user_features.iloc[user_idx].values
                    else:
                        user_feat = self.user_features[user_idx]
                        
                    if isinstance(self.item_features, pd.DataFrame):
                        item_feat = self.item_features.iloc[item_idx].values
                    else:
                        item_feat = self.item_features[item_idx]
                    
                    input_data = np.concatenate([user_feat, item_feat])
                else:
                    self.logger.error("Cannot prepare input data: features not provided")
                    return None
            
            # Get feature contributions
            if hasattr(self.base_model, 'feature_importances_'):
                # For tree-based models
                importances = self.base_model.feature_importances_
                feature_names = self._get_feature_names()
                
                # Create DataFrame
                contributions = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
            elif self.shap_explainer is not None:
                # Use SHAP values
                shap_values = self.explain_with_shap(input_data)
                
                if shap_values is None:
                    return None
                    
                feature_names = self._get_feature_names()
                
                # Create DataFrame
                if isinstance(shap_values, list):
                    # For multi-output models, use the first output
                    contributions = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.abs(shap_values[0]).mean(0)
                    })
                else:
                    contributions = pd.DataFrame({
                        'feature': feature_names,
                        'importance': np.abs(shap_values).mean(0)
                    })
            else:
                return None
            
            # Sort by importance and get top features
            contributions = contributions.sort_values('importance', ascending=False).head(top_n)
            
            return contributions
        except Exception as e:
            self.logger.error(f"Error explaining by feature contribution: {str(e)}")
            return None
    
    def generate_natural_language_explanation(self, user_id, item_id, explanation_type='all'):
        """
        Generate a natural language explanation for a recommendation.
        
        Args:
            user_id: ID of the user
            item_id: ID of the recommended item
            explanation_type: Type of explanation ('similar_items', 'user_history', 'feature', or 'all')
            
        Returns:
            String with natural language explanation
        """
        explanations = []
        item_name = self._get_item_name(item_id)
        
        try:
            # Similar items explanation
            if explanation_type in ['similar_items', 'all']:
                similar_items = self.explain_by_similar_items(item_id, n_similar=3)
                if similar_items is not None and len(similar_items) > 0:
                    similar_names = [self._get_item_name(i) for i in similar_items['item_id']]
                    similar_text = ", ".join(similar_names)
                    explanations.append(f"We recommend {item_name} because it is similar to {similar_text} which match your interests.")
            
            # User history explanation
            if explanation_type in ['user_history', 'all']:
                history_items = self.explain_by_user_history(user_id, item_id, n_items=3)
                if history_items is not None and len(history_items) > 0:
                    history_names = [self._get_item_name(i) for i in history_items['item_id']]
                    history_text = ", ".join(history_names)
                    explanations.append(f"Based on your high ratings for {history_text}, we think you'll enjoy {item_name}.")
            
            # Feature contribution explanation
            if explanation_type in ['feature', 'all']:
                features = self.explain_by_feature_contribution(user_id, item_id, top_n=3)
                if features is not None and len(features) > 0:
                    feature_text = ", ".join(features['feature'].tolist())
                    explanations.append(f"The most important factors in this recommendation were: {feature_text}.")
            
            # Add item-specific explanation if metadata is available
            if self.item_metadata is not None:
                item_meta = self._get_item_metadata(item_id)
                if item_meta is not None:
                    # Example: use genre or category information
                    if 'genre' in item_meta:
                        explanations.append(f"{item_name} is in the {item_meta['genre']} genre, which seems to match your preferences.")
                    elif 'category' in item_meta:
                        explanations.append(f"{item_name} is in the {item_meta['category']} category, which aligns with your interests.")
            
            # Combine explanations
            if explanations:
                return " ".join(explanations)
            else:
                return f"We recommend {item_name} based on your overall preference patterns."
        except Exception as e:
            self.logger.error(f"Error generating natural language explanation: {str(e)}")
            return f"We recommend {item_name} based on your preferences."
    
    def visualize_explanation(self, user_id, item_id, plot_type='radar', save_path=None):
        """
        Generate a visualization to explain a recommendation.
        
        Args:
            user_id: ID of the user
            item_id: ID of the recommended item
            plot_type: Type of plot ('radar', 'bar', or 'similarity')
            save_path: Path to save the visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        try:
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'radar':
                # Radar chart of feature contributions
                features = self.explain_by_feature_contribution(user_id, item_id, top_n=5)
                if features is not None and len(features) > 0:
                    # Prepare data for radar chart
                    categories = features['feature'].tolist()
                    values = features['importance'].tolist()
                    
                    # Create radar chart
                    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                    values += values[:1]  # Close the loop
                    angles += angles[:1]  # Close the loop
                    categories += categories[:1]  # Close the loop
                    
                    ax = plt.subplot(111, polar=True)
                    ax.plot(angles, values, 'o-', linewidth=2)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                    ax.set_title(f"Feature Importance for Recommending Item {item_id}")
                    
            elif plot_type == 'bar':
                # Bar chart of feature contributions
                features = self.explain_by_feature_contribution(user_id, item_id, top_n=10)
                if features is not None and len(features) > 0:
                    sns.barplot(x='importance', y='feature', data=features)
                    plt.title(f"Feature Importance for Recommending Item {item_id}")
                    plt.tight_layout()
                    
            elif plot_type == 'similarity':
                # Similarity to items in user history
                history_items = self.explain_by_user_history(user_id, item_id, n_items=10)
                if history_items is not None and len(history_items) > 0:
                    if 'relevance' in history_items.columns:
                        # Plot relevance and rating
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        
                        # Bar chart for ratings
                        ax1.set_xlabel('Item ID')
                        ax1.set_ylabel('Rating', color='tab:blue')
                        ax1.bar(history_items['item_id'].astype(str), history_items['rating'], 
                               color='tab:blue', alpha=0.7)
                        ax1.tick_params(axis='y', labelcolor='tab:blue')
                        
                        # Line plot for relevance
                        ax2 = ax1.twinx()
                        ax2.set_ylabel('Relevance to Recommended Item', color='tab:red')
                        ax2.plot(history_items['item_id'].astype(str), history_items['relevance'], 
                                'o-', color='tab:red')
                        ax2.tick_params(axis='y', labelcolor='tab:red')
                        
                        plt.title(f"User {user_id}'s History and Relevance to Item {item_id}")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                    else:
                        # Just plot ratings
                        sns.barplot(x='item_id', y='rating', data=history_items)
                        plt.title(f"User {user_id}'s Ratings Relevant to Item {item_id}")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
            
            return plt.gcf()
        except Exception as e:
            self.logger.error(f"Error visualizing explanation: {str(e)}")
            return None
    
    def _get_user_index(self, user_id):
        """
        Get the index for a user ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User index or None if not found
        """
        # Override in subclasses for specific implementations
        if hasattr(self, 'user_id_map'):
            return self.user_id_map.get(user_id)
        elif isinstance(self.user_features, pd.DataFrame) and 'user_id' in self.user_features.columns:
            try:
                return self.user_features[self.user_features['user_id'] == user_id].index[0]
            except IndexError:
                self.logger.warning(f"User ID {user_id} not found")
                return None
        elif isinstance(self.user_item_matrix, pd.DataFrame):
            try:
                return self.user_item_matrix.index.get_loc(user_id)
            except KeyError:
                self.logger.warning(f"User ID {user_id} not found")
                return None
        else:
            # Default implementation - assumes user_id is already an index
            return user_id
    
    def _get_item_index(self, item_id):
        """
        Get the index for an item ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item index or None if not found
        """
        # Override in subclasses for specific implementations
        if hasattr(self, 'item_id_map'):
            return self.item_id_map.get(item_id)
        elif isinstance(self.item_features, pd.DataFrame) and 'item_id' in self.item_features.columns:
            try:
                return self.item_features[self.item_features['item_id'] == item_id].index[0]
            except IndexError:
                self.logger.warning(f"Item ID {item_id} not found")
                return None
        elif isinstance(self.user_item_matrix, pd.DataFrame):
            try:
                return self.user_item_matrix.columns.get_loc(item_id)
            except KeyError:
                self.logger.warning(f"Item ID {item_id} not found")
                return None
        else:
            # Default implementation - assumes item_id is already an index
            return item_id
    
    def _get_item_ids(self, indices):
        """
        Convert item indices to item IDs.
        
        Args:
            indices: Array of item indices
            
        Returns:
            Array of item IDs
        """
        # Override in subclasses for specific implementations
        if hasattr(self, 'item_id_map'):
            reverse_map = {v: k for k, v in self.item_id_map.items()}
            return [reverse_map.get(idx) for idx in indices]
        elif isinstance(self.item_features, pd.DataFrame) and 'item_id' in self.item_features.columns:
            return self.item_features.iloc[indices]['item_id'].values
        elif isinstance(self.user_item_matrix, pd.DataFrame):
            return [self.user_item_matrix.columns[idx] for idx in indices]
        else:
            # Default implementation - return indices as IDs
            return indices
    
    def _get_item_name(self, item_id):
        """
        Get the name of an item.
        
        Args:
            item_id: Item ID
            
        Returns:
            Item name or item ID as string if name not available
        """
        if self.item_metadata is not None:
            try:
                if isinstance(self.item_metadata, pd.DataFrame):
                    if 'name' in self.item_metadata.columns:
                        return self.item_metadata[self.item_metadata['item_id'] == item_id]['name'].values[0]
                    elif 'title' in self.item_metadata.columns:
                        return self.item_metadata[self.item_metadata['item_id'] == item_id]['title'].values[0]
            except (IndexError, KeyError):
                pass
        
        # Default to item ID if name not available
        return str(item_id)
    
    def _get_item_metadata(self, item_id):
        """
        Get metadata for an item.
        
        Args:
            item_id: Item ID
            
        Returns:
            Dictionary of item metadata or None if not available
        """
        if self.item_metadata is not None:
            try:
                if isinstance(self.item_metadata, pd.DataFrame):
                    item_data = self.item_metadata[self.item_metadata['item_id'] == item_id]
                    if len(item_data) > 0:
                        return item_data.iloc[0].to_dict()
            except (IndexError, KeyError):
                pass
        
        return None
    
    def _get_feature_names(self):
        """
        Get feature names for the model.
        
        Returns:
            List of feature names
        """
        # Override in subclasses for specific implementations
        if hasattr(self, 'feature_names'):
            return self.feature_names
        elif hasattr(self.base_model, 'feature_names_in_'):
            return self.base_model.feature_names_in_
        else:
            # Generate generic feature names
            n_features = 0
            if hasattr(self.base_model, 'n_features_in_'):
                n_features = self.base_model.n_features_in_
            elif hasattr(self.base_model, 'feature_importances_'):
                n_features = len(self.base_model.feature_importances_)
            elif self.user_features is not None and self.item_features is not None:
                if isinstance(self.user_features, pd.DataFrame):
                    user_dim = self.user_features.shape[1]
                else:
                    user_dim = self.user_features.shape[1]
                    
                if isinstance(self.item_features, pd.DataFrame):
                    item_dim = self.item_features.shape[1]
                else:
                    item_dim = self.item_features.shape[1]
                    
                n_features = user_dim + item_dim
            
            return [f"Feature_{i}" for i in range(n_features)]


class ExplainableNCF(ExplainableRecommender):
    """
    Explainable Neural Collaborative Filtering model that extends the base
    ExplainableRecommender with NCF-specific functionality.
    """
    
    def __init__(self, ncf_model, user_item_df, item_metadata=None, user_encoder=None, item_encoder=None):
        """
        Initialize the explainable NCF model.
        
        Args:
            ncf_model: Trained Neural Collaborative Filtering model
            user_item_df: DataFrame with user-item interactions
            item_metadata: DataFrame with item metadata
            user_encoder: Encoder for user IDs
            item_encoder: Encoder for item IDs
        """
        # Create user-item matrix from DataFrame
        user_item_matrix = user_item_df.pivot(
            index='user_id', columns='item_id', values='rating').fillna(0)
        
        super().__init__(
            base_model=ncf_model,
            user_item_matrix=user_item_matrix,
            item_metadata=item_metadata
        )
        
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        
        # Create ID mappings
        self.user_id_map = {}
        if user_encoder is not None:
            for i, user_id in enumerate(user_encoder.classes_):
                self.user_id_map[user_id] = i
        
        self.item_id_map = {}
        if item_encoder is not None:
            for i, item_id in enumerate(item_encoder.classes_):
                self.item_id_map[item_id] = i
    
    def _prepare_model_input(self, user_idx, item_idx):
        """
        Prepare input for the NCF model.
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Model input
        """
        # For NCF, input is typically [user_id, item_id]
        return [np.array([[user_idx]]), np.array([[item_idx]])]
    
    def explain_recommendation(self, user_id, item_id, n_similar=5, explanation_type='all'):
        """
        Generate a comprehensive explanation for an NCF recommendation.
        
        Args:
            user_id: User ID
            item_id: Item ID of the recommended item
            n_similar: Number of similar items to include
            explanation_type: Type of explanation
            
        Returns:
            Dictionary with explanation components
        """
        # Encode IDs if encoders are available
        if self.user_encoder is not None:
            try:
                user_idx = self.user_encoder.transform([user_id])[0]
            except ValueError:
                self.logger.warning(f"User ID {user_id} not found in encoder")
                user_idx = user_id
        else:
            user_idx = user_id
            
        if self.item_encoder is not None:
            try:
                item_idx = self.item_encoder.transform([item_id])[0]
            except ValueError:
                self.logger.warning(f"Item ID {item_id} not found in encoder")
                item_idx = item_id
        else:
            item_idx = item_id
        
        # Get item metadata
        item_info = self._get_item_metadata(item_id)
        
        # Get similar items the user might like
        similar_items = self.explain_by_similar_items(item_id, n_similar=n_similar)
        
        # Get relevant items from user history
        history_items = self.explain_by_user_history(user_id, item_id, n_items=n_similar)
        
        # Generate natural language explanation
        text_explanation = self.generate_natural_language_explanation(
            user_id, item_id, explanation_type=explanation_type)
        
        # Create visualization
        if similar_items is not None and len(similar_items) > 0:
            visualization = self.visualize_explanation(user_id, item_id, plot_type='similarity')
        else:
            visualization = None
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'item_info': item_info,
            'similar_items': similar_items,
            'user_history': history_items,
            'text_explanation': text_explanation,
            'visualization': visualization
        }


class ExplainableGNN(ExplainableRecommender):
    """
    Explainable Graph Neural Network model that extends the base
    ExplainableRecommender with GNN-specific functionality.
    """
    
    def __init__(self, gnn_model, user_item_df, item_metadata=None, 
                 user_encoder=None, item_encoder=None, graph=None):
        """
        Initialize the explainable GNN model.
        
        Args:
            gnn_model: Trained Graph Neural Network model
            user_item_df: DataFrame with user-item interactions
            item_metadata: DataFrame with item metadata
            user_encoder: Encoder for user IDs
            item_encoder: Encoder for item IDs
            graph: NetworkX graph used by the GNN model
        """
        # Create user-item matrix from DataFrame
        user_item_matrix = user_item_df.pivot(
            index='user_id', columns='item_id', values='rating').fillna(0)
        
        super().__init__(
            base_model=gnn_model,
            user_item_matrix=user_item_matrix,
            item_metadata=item_metadata
        )
        
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.graph = graph
        
        # Create ID mappings
        self.user_id_map = {}
        if user_encoder is not None:
            for i, user_id in enumerate(user_encoder.classes_):
                self.user_id_map[user_id] = i
        
        self.item_id_map = {}
        if item_encoder is not None:
            for i, item_id in enumerate(item_encoder.classes_):
                self.item_id_map[item_id] = i
    
    def explain_by_graph_paths(self, user_id, item_id, max_path_length=3, n_paths=5):
        """
        Explain a recommendation by showing paths in the graph connecting the user to the item.
        
        Args:
            user_id: User ID
            item_id: Item ID
            max_path_length: Maximum path length to consider
            n_paths: Maximum number of paths to return
            
        Returns:
            List of paths connecting the user to the item
        """
        if self.graph is None:
            self.logger.error("Cannot explain by graph paths: graph not provided")
            return None
        
        import networkx as nx
        
        try:
            # Convert IDs to node names if necessary
            user_node = f"user_{user_id}"
            item_node = f"item_{item_id}"
            
            # Check if nodes exist in the graph
            if user_node not in self.graph or item_node not in self.graph:
                self.logger.warning(f"User {user_node} or item {item_node} not in graph")
                return None
            
            # Find all simple paths up to max_path_length
            paths = list(nx.all_simple_paths(
                self.graph, source=user_node, target=item_node, cutoff=max_path_length))
            
            # Sort paths by length (shorter paths first)
            paths.sort(key=len)
            
            # Return at most n_paths
            return paths[:n_paths]
        except Exception as e:
            self.logger.error(f"Error finding graph paths: {str(e)}")
            return None
    
    def explain_by_node_importance(self, user_id, item_id, n_nodes=10):
        """
        Explain a recommendation by showing the most important nodes in the graph.
        
        Args:
            user_id: User ID
            item_id: Item ID
            n_nodes: Number of important nodes to include
            
        Returns:
            DataFrame of important nodes with scores
        """
        if self.graph is None:
            self.logger.error("Cannot explain by node importance: graph not provided")
            return None
        
        import networkx as nx
        
        try:
            # Convert IDs to node names if necessary
            user_node = f"user_{user_id}"
            item_node = f"item_{item_id}"
            
            # Check if nodes exist in the graph
            if user_node not in self.graph or item_node not in self.graph:
                self.logger.warning(f"User {user_node} or item {item_node} not in graph")
                return None
            
            # Calculate node centrality measures
            betweenness = nx.betweenness_centrality(self.graph)
            pagerank = nx.pagerank(self.graph)
            
            # Combine centrality measures
            node_importance = {}
            for node in self.graph.nodes():
                # Skip the user and item nodes themselves
                if node == user_node or node == item_node:
                    continue
                    
                # Calculate combined importance score
                importance = 0.5 * betweenness.get(node, 0) + 0.5 * pagerank.get(node, 0)
                node_importance[node] = importance
            
            # Sort nodes by importance
            sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Create DataFrame
            important_nodes = pd.DataFrame(sorted_nodes[:n_nodes], columns=['node', 'importance'])
            
            # Add node type and name
            important_nodes['type'] = important_nodes['node'].apply(
                lambda x: x.split('_')[0])
            important_nodes['id'] = important_nodes['node'].apply(
                lambda x: '_'.join(x.split('_')[1:]))
            
            # Add item metadata if available
            if self.item_metadata is not None:
                # Only join for item nodes
                item_nodes = important_nodes[important_nodes['type'] == 'item'].copy()
                if len(item_nodes) > 0:
                    item_nodes = item_nodes.merge(
                        self.item_metadata, left_on='id', right_on='item_id', how='left')
                    
                    # Update the original DataFrame
                    for col in self.item_metadata.columns:
                        if col != 'item_id':
                            important_nodes.loc[important_nodes['type'] == 'item', col] = \
                                item_nodes[col].values
            
            return important_nodes
        except Exception as e:
            self.logger.error(f"Error calculating node importance: {str(e)}")
            return None
    
    def explain_recommendation(self, user_id, item_id, n_similar=5, include_paths=True):
        """
        Generate a comprehensive explanation for a GNN recommendation.
        
        Args:
            user_id: User ID
            item_id: Item ID of the recommended item
            n_similar: Number of similar items to include
            include_paths: Whether to include graph paths in the explanation
            
        Returns:
            Dictionary with explanation components
        """
        # Get item metadata
        item_info = self._get_item_metadata(item_id)
        
        # Get similar items the user might like
        similar_items = self.explain_by_similar_items(item_id, n_similar=n_similar)
        
        # Get relevant items from user history
        history_items = self.explain_by_user_history(user_id, item_id, n_items=n_similar)
        
        # Get graph-specific explanations
        if include_paths:
            graph_paths = self.explain_by_graph_paths(user_id, item_id, max_path_length=3, n_paths=3)
        else:
            graph_paths = None
            
        important_nodes = self.explain_by_node_importance(user_id, item_id, n_nodes=5)
        
        # Generate natural language explanation
        text_explanation = self.generate_natural_language_explanation(user_id, item_id)
        
        # Create visualization
        visualization = self.visualize_explanation(user_id, item_id, plot_type='similarity')
        
        return {
            'user_id': user_id,
            'item_id': item_id,
            'item_info': item_info,
            'similar_items': similar_items,
            'user_history': history_items,
            'graph_paths': graph_paths,
            'important_nodes': important_nodes,
            'text_explanation': text_explanation,
            'visualization': visualization
        }


def create_explainable_recommender(model_type, model, ratings_df, item_metadata_df=None, 
                                 user_encoder=None, item_encoder=None, graph=None):
    """
    Create an explainable recommender based on the model type.
    
    Args:
        model_type: Type of model ('ncf', 'gnn', or 'generic')
        model: Trained recommendation model
        ratings_df: DataFrame with user-item ratings
        item_metadata_df: DataFrame with item metadata (optional)
        user_encoder: Encoder for user IDs (optional)
        item_encoder: Encoder for item IDs (optional)
        graph: NetworkX graph for GNN models (optional)
        
    Returns:
        Explainable recommender instance
    """
    if model_type.lower() == 'ncf':
        return ExplainableNCF(
            ncf_model=model,
            user_item_df=ratings_df,
            item_metadata=item_metadata_df,
            user_encoder=user_encoder,
            item_encoder=item_encoder
        )
    elif model_type.lower() == 'gnn':
        return ExplainableGNN(
            gnn_model=model,
            user_item_df=ratings_df,
            item_metadata=item_metadata_df,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            graph=graph
        )
    else:
        # Generic explainable recommender
        user_item_matrix = ratings_df.pivot(
            index='user_id', columns='item_id', values='rating').fillna(0)
        
        return ExplainableRecommender(
            base_model=model,
            user_item_matrix=user_item_matrix,
            item_metadata=item_metadata_df
        )


def get_recommendation_explanation(explainer, user_id, item_id, format_type='text'):
    """
    Get an explanation for a recommendation in the specified format.
    
    Args:
        explainer: Explainable recommender instance
        user_id: User ID
        item_id: Item ID of the recommended item
        format_type: Type of explanation format ('text', 'visual', or 'full')
        
    Returns:
        Explanation in the specified format
    """
    if format_type == 'text':
        return explainer.generate_natural_language_explanation(user_id, item_id)
    elif format_type == 'visual':
        return explainer.visualize_explanation(user_id, item_id)
    elif format_type == 'full':
        # Check if the explainer has a model-specific explanation method
        if hasattr(explainer, 'explain_recommendation'):
            return explainer.explain_recommendation(user_id, item_id)
        else:
            # Create a comprehensive explanation manually
            explanation = {
                'user_id': user_id,
                'item_id': item_id,
                'item_info': explainer._get_item_metadata(item_id),
                'similar_items': explainer.explain_by_similar_items(item_id),
                'user_history': explainer.explain_by_user_history(user_id, item_id),
                'text_explanation': explainer.generate_natural_language_explanation(user_id, item_id),
                'visualization': explainer.visualize_explanation(user_id, item_id)
            }
            return explanation
    else:
        raise ValueError(f"Unknown format type: {format_type}")