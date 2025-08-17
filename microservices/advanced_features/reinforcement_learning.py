import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import random
from collections import deque
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler


def create_rl_recommender(users_df=None, items_df=None, interactions_df=None):
    """
    Create a reinforcement learning recommender system
    
    Args:
        users_df: DataFrame with user information
        items_df: DataFrame with item information
        interactions_df: DataFrame with user-item interactions
        
    Returns:
        Dictionary with recommender and related data
    """
    # For demo purposes, create a simple mock environment if no data is provided
    if users_df is None or items_df is None or interactions_df is None:
        # Create mock data
        n_users = 100
        n_items = 200
        
        # Create mock user-item matrix
        user_item_matrix = np.zeros((n_users, n_items))
        for i in range(n_users):
            # Each user has interacted with 5-15 random items
            n_interactions = np.random.randint(5, 15)
            item_indices = np.random.choice(n_items, n_interactions, replace=False)
            user_item_matrix[i, item_indices] = np.random.uniform(1, 5, n_interactions)
        
        # Create mock item features
        item_features_matrix = np.random.random((n_items, 10))  # 10 features per item
        
        # Create mock user features
        user_features_matrix = np.random.random((n_users, 8))  # 8 features per user
        
        # Create mock encoders
        user_encoder = LabelEncoder()
        user_encoder.fit(np.arange(n_users))
        
        item_encoder = LabelEncoder()
        item_encoder.fit(np.arange(n_items))
    else:
        # Process real data
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()
        
        # Encode user and item IDs
        user_ids = user_encoder.fit_transform(users_df['user_id'].unique())
        item_ids = item_encoder.fit_transform(items_df['item_id'].unique())
        
        # Create user-item matrix
        n_users = len(user_ids)
        n_items = len(item_ids)
        user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions_df.iterrows():
            user_idx = user_encoder.transform([row['user_id']])[0]
            item_idx = item_encoder.transform([row['item_id']])[0]
            user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Create item features matrix
        if 'category' in items_df.columns:
            # One-hot encode categorical features
            item_categories = pd.get_dummies(items_df['category'])
            item_features_matrix = np.hstack([
                items_df[['price']].values,  # Numerical features
                item_categories.values  # One-hot encoded categories
            ])
        else:
            # Use only numerical features if available
            numerical_cols = items_df.select_dtypes(include=[np.number]).columns
            item_features_matrix = items_df[numerical_cols].values
        
        # Create user features matrix
        if 'age' in users_df.columns and 'gender' in users_df.columns:
            # One-hot encode categorical features
            user_gender = pd.get_dummies(users_df['gender'])
            user_features_matrix = np.hstack([
                users_df[['age']].values,  # Numerical features
                user_gender.values  # One-hot encoded gender
            ])
        else:
            # Use only numerical features if available
            numerical_cols = users_df.select_dtypes(include=[np.number]).columns
            user_features_matrix = users_df[numerical_cols].values
        
        # Normalize numerical features
        scaler = StandardScaler()
        user_features_matrix = scaler.fit_transform(user_features_matrix)
    
    # Create the RL recommender
    recommender = RLRecommender(
        user_item_matrix=user_item_matrix,
        item_features=item_features_matrix,
        user_features=user_features_matrix,
        hidden_size=128,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=10
    )
    
    return recommender

class RecommenderEnvironment:
    """
    Environment for reinforcement learning-based recommendation system.
    
    This environment simulates user interactions with recommendations and provides
    rewards based on user engagement metrics (clicks, views, purchases, etc.).
    """
    
    def __init__(self, user_item_matrix, item_features=None, user_features=None, 
                 reward_weights={'click': 1.0, 'view_time': 0.5, 'purchase': 5.0}):
        """
        Initialize the recommender environment.
        """
        self.user_item_matrix = user_item_matrix
        self.item_features = item_features
        self.user_features = user_features
        self.reward_weights = reward_weights
        self.n_users, self.n_items = user_item_matrix.shape
        self.current_user = None
        self.recommended_items = []
        self.timestep = 0
        self.max_timesteps = 20  # Maximum number of interactions per episode
        # Logger setup
        self.logger = logging.getLogger(__name__)
        # Add observation_space attribute
        dummy_state = self._get_dummy_state()
        self.observation_space = type('Space', (), {'shape': dummy_state.shape})()
        # Add action_space attribute
        self.action_space = type('Space', (), {'n': self.n_items})()

    def _get_dummy_state(self):
        # Helper to get the shape of the state vector
        user_profile = np.zeros(self.n_items)
        if self.user_features is not None:
            user_feature_vector = np.zeros(self.user_features.shape[1])
        else:
            user_feature_vector = np.array([])
        history = np.zeros(self.n_items)
        state = np.concatenate([user_profile, history, user_feature_vector])
        return state

    def reset(self, user_id=None):
        """
        Reset the environment for a new episode.
        
        Args:
            user_id: Specific user ID to reset for, or random if None
            
        Returns:
            Initial state representation
        """
        # Select a random user if not specified
        if user_id is None:
            self.current_user = np.random.randint(0, self.n_users)
        else:
            self.current_user = user_id
            
        self.recommended_items = []
        self.timestep = 0
        
        # Return the initial state
        return self._get_state()
    
    def _get_state(self):
        """
        Get the current state representation.
        
        Returns:
            State vector combining user profile, interaction history, and context
        """
        # User profile (from ratings or features)
        user_profile = self.user_item_matrix[self.current_user].copy()
        
        # Add user features if available
        if self.user_features is not None:
            user_feature_vector = self.user_features[self.current_user]
        else:
            user_feature_vector = np.array([])
        
        # Recent interaction history (last 5 recommended items)
        history = np.zeros(self.n_items)
        for item in self.recommended_items[-5:]:
            if item < self.n_items:
                history[item] = 1
        
        # Combine all information into state vector
        state = np.concatenate([user_profile, history, user_feature_vector])
        
        return state
    
    def step(self, action):
        """
        Take a step in the environment by recommending an item.
        
        Args:
            action: Item ID to recommend
            
        Returns:
            next_state: Next state representation
            reward: Reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        # Validate action
        if action >= self.n_items:
            self.logger.warning(f"Invalid action: {action}, n_items: {self.n_items}")
            return self._get_state(), -1.0, False, {"error": "Invalid action"}
        
        # Check if item was already recommended
        if action in self.recommended_items:
            reward = -0.5  # Penalty for recommending the same item again
        else:
            # Calculate reward based on user's preference and engagement
            base_reward = self.user_item_matrix[self.current_user, action]
            
            # Simulate user engagement (click, view time, purchase)
            engagement = self._simulate_user_engagement(action)
            
            # Calculate weighted reward
            reward = base_reward
            for eng_type, value in engagement.items():
                if eng_type in self.reward_weights:
                    reward += value * self.reward_weights[eng_type]
        
        # Update state
        self.recommended_items.append(action)
        self.timestep += 1
        
        # Check if episode is done
        done = self.timestep >= self.max_timesteps
        
        # Get next state
        next_state = self._get_state()
        
        # Additional info
        info = {
            "timestep": self.timestep,
            "user_id": self.current_user,
            "item_id": action
        }
        
        return next_state, reward, done, info
    
    def _simulate_user_engagement(self, item_id):
        """
        Simulate user engagement with the recommended item.
        
        Args:
            item_id: ID of the recommended item
            
        Returns:
            Dictionary of engagement metrics
        """
        # Base probability of engagement based on user-item rating
        base_prob = max(0.1, self.user_item_matrix[self.current_user, item_id])
        
        # Simulate different types of engagement
        click = 1 if np.random.random() < base_prob else 0
        
        # Only simulate view time and purchase if clicked
        view_time = 0
        purchase = 0
        if click == 1:
            # View time (normalized between 0 and 1)
            view_time = np.random.beta(2, 5) if base_prob > 0.3 else np.random.beta(1, 10)
            
            # Purchase (rare event)
            purchase = 1 if np.random.random() < (base_prob * 0.3) else 0
        
        return {
            "click": click,
            "view_time": view_time,
            "purchase": purchase
        }


class DQNAgent:
    """
    Deep Q-Network agent for reinforcement learning-based recommendations.
    
    This agent learns to recommend items that maximize long-term cumulative rewards
    by using a deep Q-network to approximate the action-value function.
    """
    
    def __init__(self, state_size, action_size, hidden_size=128, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of the state space
            action_size: Dimension of the action space (number of items)
            hidden_size: Size of hidden layers in the Q-network
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of the replay memory
            batch_size: Batch size for training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build the Q-network and target network
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.update_target_network()
    
    def _build_model(self):
        """
        Build the deep Q-network model.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.hidden_size, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_network(self):
        """
        Update the target network with weights from the Q-network.
        """
        self.target_network.set_weights(self.q_network.get_weights())
        self.logger.debug("Target network updated")
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state
            valid_actions: List of valid actions (optional)
            
        Returns:
            Chosen action
        """
        # Default to all actions being valid if not specified
        if valid_actions is None:
            valid_actions = list(range(self.action_size))
        
        # Explore: choose a random action
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        # Exploit: choose the best action based on Q-values
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        
        # Filter Q-values for valid actions only
        valid_q = {a: q_values[a] for a in valid_actions}
        
        # Return action with highest Q-value among valid actions
        return max(valid_q, key=valid_q.get)
    
    def replay(self, batch_size=None):
        """
        Train the model using experience replay.
        
        Args:
            batch_size: Size of the batch to train on (default: self.batch_size)
            
        Returns:
            Loss value from training
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Check if we have enough samples in memory
        if len(self.memory) < batch_size:
            return 0
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            # Current Q-values
            target = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                # If episode is done, there is no future reward
                target[action] = reward
            else:
                # Double DQN: Use Q-network to select action, target network to evaluate it
                next_action = np.argmax(self.q_network.predict(next_state.reshape(1, -1), verbose=0)[0])
                next_q = self.target_network.predict(next_state.reshape(1, -1), verbose=0)[0][next_action]
                
                # Update target with reward and discounted future reward
                target[action] = reward + self.gamma * next_q
            
            targets[i] = target
        
        # Train the model
        history = self.q_network.fit(states, targets, epochs=1, verbose=0, batch_size=batch_size)
        loss = history.history['loss'][0]
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def load(self, filepath):
        """
        Load model weights from file.
        
        Args:
            filepath: Path to the saved model
        """
        self.q_network.load_weights(filepath)
        self.update_target_network()
        self.logger.info(f"Model loaded from {filepath}")
    
    def save(self, filepath):
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save the model
        """
        self.q_network.save_weights(filepath)
        self.logger.info(f"Model saved to {filepath}")


class RLRecommender:
    """
    Reinforcement Learning-based Recommender System.
    This system uses a DQN agent to learn optimal recommendation policies by interacting with a simulated environment.
    """
    def __init__(self, user_item_matrix, item_features=None, user_features=None,
                 hidden_size=128, learning_rate=0.001, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 memory_size=10000, batch_size=32, target_update_freq=10):
        self.env = RecommenderEnvironment(
            user_item_matrix=user_item_matrix,
            item_features=item_features,
            user_features=user_features
        )
        self.agent = DQNAgent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.n,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            memory_size=memory_size,
            batch_size=batch_size
        )
        self.target_update_freq = target_update_freq
        self.episode_count = 0
        self.logger = logging.getLogger(__name__)
    def train(self, n_episodes=1000, max_steps=20, verbose=1, eval_freq=100):
        self.logger.info(f"Starting training for {n_episodes} episodes")
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'eval_rewards': []
        }
        for episode in range(n_episodes):
            state = self.env.reset()
            total_reward = 0
            step_count = 0
            episode_losses = []
            for step in range(max_steps):
                valid_actions = [i for i in range(self.env.n_items) if i not in self.env.recommended_items]
                action = self.agent.act(state, valid_actions)
                next_state, reward, done, info = self.env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                loss = self.agent.replay()
                if loss > 0:
                    episode_losses.append(loss)
                state = next_state
                total_reward += reward
                step_count += 1
                if done:
                    break
            if episode % self.target_update_freq == 0:
                self.agent.update_target_network()
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(step_count)
            metrics['losses'].append(np.mean(episode_losses) if episode_losses else 0)
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(n_episodes=5, max_steps=max_steps)
                metrics['eval_rewards'].append(eval_reward)
                if verbose > 0:
                    self.logger.info(f"Episode {episode}/{n_episodes}, Reward: {total_reward:.2f}, Eval Reward: {eval_reward:.2f}, Epsilon: {self.agent.epsilon:.4f}")
            elif verbose > 1:
                self.logger.debug(f"Episode {episode}/{n_episodes}, Reward: {total_reward:.2f}, Steps: {step_count}")
            self.episode_count += 1
        self.logger.info("Training completed")
        return metrics
    def evaluate(self, n_episodes=10, max_steps=20):
        total_rewards = []
        current_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.05
        for episode in range(n_episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                valid_actions = [i for i in range(self.env.n_items) if i not in self.env.recommended_items]
                action = self.agent.act(state, valid_actions)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
                if done:
                    break
            total_rewards.append(episode_reward)
        self.agent.epsilon = current_epsilon
        return np.mean(total_rewards)
    def recommend(self, user_id, n_recommendations=5, exclude_rated=True):
        state = self.env.reset(user_id)
        recommendations = []
        if exclude_rated:
            rated_items = np.where(self.env.user_item_matrix[user_id] > 0)[0]
        else:
            rated_items = []
        for _ in range(n_recommendations):
            valid_actions = [i for i in range(self.env.n_items) if i not in recommendations and i not in rated_items]
            if not valid_actions:
                break
            self.agent.epsilon = 0.0
            action = self.agent.act(state, valid_actions)
            recommendations.append(action)
            next_state, _, _, _ = self.env.step(action)
            state = next_state
        return recommendations
    def save(self, filepath):
        self.agent.save(filepath)
    def load(self, filepath):
        self.agent.load(filepath)


def create_rl_recommender(ratings_df, item_features_df=None, user_features_df=None):
    """
    Create and initialize a reinforcement learning recommender from dataframes.
    
    Args:
        ratings_df: DataFrame with columns 'user_id', 'item_id', 'rating'
        item_features_df: DataFrame with item features (optional)
        user_features_df: DataFrame with user features (optional)
        
    Returns:
        Initialized RL recommender
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import pandas as pd
    
    # Encode user and item IDs
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # Verify column names match the input data
    required_columns = ['user_id', 'item_id', 'rating']
    if not all(col in ratings_df.columns for col in ['user_id', 'item_id', 'rating']):
        raise ValueError("Missing required columns")
    
    user_ids = user_encoder.fit_transform(ratings_df['user_id'].values)
    item_ids = item_encoder.fit_transform(ratings_df['item_id'].values)
    
    # Create user-item matrix
    n_users = len(user_encoder.classes_)
    n_items = len(item_encoder.classes_)
    
    user_item_matrix = np.zeros((n_users, n_items))
    for i, row in enumerate(zip(user_ids, item_ids, ratings_df['rating'].values)):
        user_idx, item_idx, rating = row
        user_item_matrix[user_idx, item_idx] = rating / 5.0  # Normalize to [0, 1]
    
    # Process item features if available
    item_features_matrix = None
    if item_features_df is not None:
        # Ensure item_features_df has the same items as in ratings
        common_items = set(item_encoder.classes_).intersection(set(item_features_df.index))
        if len(common_items) < len(item_encoder.classes_):
            print(f"Warning: Only {len(common_items)} out of {len(item_encoder.classes_)} items have features")
        
        # Extract and normalize features
        from sklearn.preprocessing import LabelEncoder
        
        feature_cols = [col for col in item_features_df.columns if col != 'item_id']
        if feature_cols:
            # Encode categorical features
            item_features_df_encoded = item_features_df.copy()
            for col in feature_cols:
                if item_features_df[col].dtype == 'object':
                    le = LabelEncoder()
                    item_features_df_encoded[col] = le.fit_transform(item_features_df[col])
            
            # Create a matrix with items in the same order as encoded item_ids
            item_features_matrix = np.zeros((n_items, len(feature_cols)))
            for i, item in enumerate(item_encoder.classes_):
                if item in item_features_df.index:
                    item_features_matrix[i] = item_features_df_encoded.loc[item, feature_cols].values
            
            # Normalize features
            scaler = StandardScaler()
            item_features_matrix = scaler.fit_transform(item_features_matrix)
    
    # Process user features if available
    user_features_matrix = None
    if user_features_df is not None:
        # Similar processing for user features
        common_users = set(user_encoder.classes_).intersection(set(user_features_df.index))
        if len(common_users) < len(user_encoder.classes_):
            print(f"Warning: Only {len(common_users)} out of {len(user_encoder.classes_)} users have features")
        
        from sklearn.preprocessing import LabelEncoder
        
        feature_cols = [col for col in user_features_df.columns if col != 'user_id']
        if feature_cols:
            # Encode categorical features
            user_features_df_encoded = user_features_df.copy()
            for col in feature_cols:
                if user_features_df[col].dtype == 'object':
                    le = LabelEncoder()
                    user_features_df_encoded[col] = le.fit_transform(user_features_df[col])
            
            user_features_matrix = np.zeros((n_users, len(feature_cols)))
            for i, user in enumerate(user_encoder.classes_):
                if user in user_features_df.index:
                    user_features_matrix[i] = user_features_df_encoded.loc[user, feature_cols].values
            
            # Normalize features
            scaler = StandardScaler()
            user_features_matrix = scaler.fit_transform(user_features_matrix)
    
    # Create the RL recommender
    recommender = RLRecommender(
        user_item_matrix=user_item_matrix,
        item_features=item_features_matrix,
        user_features=user_features_matrix,
        hidden_size=128,
        learning_rate=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=10
    )
    
    return recommender


def train_rl_recommender(recommender_data, n_episodes=1000, max_steps=20, verbose=1):
    """
    Train a reinforcement learning recommender.
    
    Args:
        recommender_data: Dictionary with recommender and encoders from create_rl_recommender
        n_episodes: Number of episodes to train for
        max_steps: Maximum number of steps per episode
        verbose: Verbosity level
        
    Returns:
        Training metrics
    """
    recommender = recommender_data['recommender']
    
    # Train the recommender
    metrics = recommender.train(
        n_episodes=n_episodes,
        max_steps=max_steps,
        verbose=verbose,
        eval_freq=100
    )
    
    return metrics


def get_rl_recommendations(recommender_data, user_id, n_recommendations=5, exclude_rated=True):
    """
    Get recommendations for a user using the RL recommender.
    
    Args:
        recommender_data: Dictionary with recommender and encoders from create_rl_recommender
        user_id: Original user ID (before encoding)
        n_recommendations: Number of recommendations to generate
        exclude_rated: Whether to exclude items the user has already rated
        
    Returns:
        List of recommended item IDs (original IDs, not encoded)
    """
    recommender = recommender_data['recommender']
    user_encoder = recommender_data['user_encoder']
    item_encoder = recommender_data['item_encoder']
    
    # Encode the user ID
    try:
        encoded_user_id = user_encoder.transform([user_id])[0]
    except ValueError:
        print(f"User ID {user_id} not found in training data")
        return []
    
    # Get recommendations
    encoded_recommendations = recommender.recommend(
        user_id=encoded_user_id,
        n_recommendations=n_recommendations,
        exclude_rated=exclude_rated
    )
    
    # Decode item IDs
    recommendations = [item_encoder.inverse_transform([item_id])[0] 
                      for item_id in encoded_recommendations]
    
    return recommendations