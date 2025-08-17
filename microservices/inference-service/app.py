import os
import json
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# Import our model implementations
import sys
sys.path.append('/app')

# Add parent directory to path to import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from neural_collaborative_filtering import NeuralCollaborativeFiltering
from graph_neural_network import GNNRecommender
from enhanced_features import EnhancedFeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Recommendation Inference Service", version="1.0.0")

# Prometheus metrics
REQUEST_COUNT = Counter('recommendation_requests_total', 'Total number of recommendation requests')
RESPONSE_TIME = Histogram('recommendation_response_time_seconds', 'Response time for recommendation requests')
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load the model')

# Model storage
MODEL_DIR = os.getenv('MODEL_DIR', '/models')
NCF_MODEL_PATH = os.path.join(MODEL_DIR, 'ncf_model')
GNN_MODEL_PATH = os.path.join(MODEL_DIR, 'gnn_model')

# Redis configuration for caching
REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_CACHE_TTL = int(os.getenv('REDIS_CACHE_TTL', '3600'))  # 1 hour

# Model instances
ncf_model = None
gnn_model = None
feature_engineering = None

# Request and response models
class UserPreferences(BaseModel):
    user_id: int
    movie_ids: Optional[List[int]] = None
    n_recommendations: int = 10
    include_metadata: bool = False
    model_type: str = "ncf"  # Options: "ncf", "gnn", "ensemble"

class MovieRecommendation(BaseModel):
    movie_id: int
    title: Optional[str] = None
    score: float
    genre: Optional[str] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]
    model_type: str
    processing_time: float

@app.on_event("startup")
def startup_event():
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Load models
    load_models()
    
    logger.info("Recommendation Inference Service started")

def load_models():
    """
    Load trained models
    """
    global ncf_model, gnn_model, feature_engineering
    
    start_time = time.time()
    
    try:
        # Load NCF model if available
        if os.path.exists(NCF_MODEL_PATH):
            logger.info(f"Loading NCF model from {NCF_MODEL_PATH}")
            # In a real implementation, we would load the model with proper parameters
            # For demonstration, we'll create a new instance
            ncf_model = NeuralCollaborativeFiltering(n_users=1000, n_items=1000)
            # ncf_model = NeuralCollaborativeFiltering.load_model(NCF_MODEL_PATH, n_users=1000, n_items=1000)
        else:
            logger.warning(f"NCF model not found at {NCF_MODEL_PATH}")
        
        # Load GNN model if available
        if os.path.exists(GNN_MODEL_PATH):
            logger.info(f"Loading GNN model from {GNN_MODEL_PATH}")
            # In a real implementation, we would load the model with proper parameters
            # For demonstration, we'll create a new instance
            gnn_model = GNNRecommender(n_users=1000, n_items=1000, embedding_dim=32)
        else:
            logger.warning(f"GNN model not found at {GNN_MODEL_PATH}")
        
        # Initialize feature engineering
        feature_engineering = EnhancedFeatureEngineering(
            user_col='user_id',
            item_col='movie_id',
            rating_col='rating'
        )
        
        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)
        logger.info(f"Models loaded in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "models_loaded": ncf_model is not None or gnn_model is not None}

@app.get("/models")
def get_available_models():
    """
    Get information about available models
    """
    models = []
    
    if ncf_model is not None:
        models.append({
            "name": "Neural Collaborative Filtering",
            "type": "ncf",
            "path": NCF_MODEL_PATH
        })
    
    if gnn_model is not None:
        models.append({
            "name": "Graph Neural Network",
            "type": "gnn",
            "path": GNN_MODEL_PATH
        })
    
    return {"available_models": models}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(preferences: UserPreferences):
    """
    Get movie recommendations for a user
    """
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    try:
        # Check if models are loaded
        if ncf_model is None and gnn_model is None:
            raise HTTPException(status_code=503, detail="No recommendation models are currently available")
        
        # Check if requested model is available
        if preferences.model_type == "ncf" and ncf_model is None:
            raise HTTPException(status_code=404, detail="NCF model is not available")
        elif preferences.model_type == "gnn" and gnn_model is None:
            raise HTTPException(status_code=404, detail="GNN model is not available")
        elif preferences.model_type == "ensemble" and (ncf_model is None or gnn_model is None):
            raise HTTPException(status_code=404, detail="Ensemble requires both NCF and GNN models")
        
        # Generate recommendations based on model type
        if preferences.model_type == "ncf":
            recommendations = generate_ncf_recommendations(preferences)
        elif preferences.model_type == "gnn":
            recommendations = generate_gnn_recommendations(preferences)
        elif preferences.model_type == "ensemble":
            recommendations = generate_ensemble_recommendations(preferences)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {preferences.model_type}")
        
        processing_time = time.time() - start_time
        RESPONSE_TIME.observe(processing_time)
        
        return RecommendationResponse(
            user_id=preferences.user_id,
            recommendations=recommendations,
            model_type=preferences.model_type,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_ncf_recommendations(preferences: UserPreferences) -> List[MovieRecommendation]:
    """
    Generate recommendations using the NCF model
    """
    # In a real implementation, we would:
    # 1. Get candidate items (movies the user hasn't rated)
    # 2. Use the NCF model to predict ratings for these items
    # 3. Return the top-N items with highest predicted ratings
    
    # For demonstration, we'll return mock recommendations
    recommendations = [
        MovieRecommendation(
            movie_id=i,
            title=f"Movie {i}",
            score=np.random.uniform(0.7, 0.99),
            genre="Action" if i % 3 == 0 else "Comedy" if i % 3 == 1 else "Drama"
        )
        for i in range(preferences.n_recommendations)
    ]
    
    # Sort by score in descending order
    recommendations.sort(key=lambda x: x.score, reverse=True)
    
    return recommendations

def generate_gnn_recommendations(preferences: UserPreferences) -> List[MovieRecommendation]:
    """
    Generate recommendations using the GNN model
    """
    # Similar to NCF, but using the GNN model
    # For demonstration, we'll return mock recommendations
    recommendations = [
        MovieRecommendation(
            movie_id=i + 100,  # Different IDs to distinguish from NCF
            title=f"Movie {i + 100}",
            score=np.random.uniform(0.7, 0.99),
            genre="Sci-Fi" if i % 3 == 0 else "Thriller" if i % 3 == 1 else "Romance"
        )
        for i in range(preferences.n_recommendations)
    ]
    
    # Sort by score in descending order
    recommendations.sort(key=lambda x: x.score, reverse=True)
    
    return recommendations

def generate_ensemble_recommendations(preferences: UserPreferences) -> List[MovieRecommendation]:
    """
    Generate recommendations using an ensemble of models
    """
    # Get recommendations from both models
    ncf_recs = generate_ncf_recommendations(preferences)
    gnn_recs = generate_gnn_recommendations(preferences)
    
    # Combine and deduplicate
    movie_scores = {}
    for rec in ncf_recs + gnn_recs:
        if rec.movie_id in movie_scores:
            # Average the scores if movie appears in both lists
            movie_scores[rec.movie_id] = (
                movie_scores[rec.movie_id][0] + rec.score,
                movie_scores[rec.movie_id][1] + 1,
                rec.title,
                rec.genre
            )
        else:
            movie_scores[rec.movie_id] = (rec.score, 1, rec.title, rec.genre)
    
    # Calculate average scores
    ensemble_recs = [
        MovieRecommendation(
            movie_id=movie_id,
            title=info[2],
            score=info[0] / info[1],  # Average score
            genre=info[3]
        )
        for movie_id, info in movie_scores.items()
    ]
    
    # Sort by score and limit to requested number
    ensemble_recs.sort(key=lambda x: x.score, reverse=True)
    return ensemble_recs[:preferences.n_recommendations]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)