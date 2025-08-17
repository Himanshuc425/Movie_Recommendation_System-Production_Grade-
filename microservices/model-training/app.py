import os
import json
import time
import logging
import mlflow
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Counter, Gauge, Summary

# Import our model implementations
import sys
sys.path.append('/app')

# Add parent directory to path to import from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from neural_collaborative_filtering import NeuralCollaborativeFiltering, train_ncf_model
from graph_neural_network import GNNRecommender
from enhanced_features import EnhancedFeatureEngineering

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Model Training Service", version="1.0.0")

# Prometheus metrics
TRAINING_DURATION = Summary('model_training_duration_seconds', 'Time taken to train a model')
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy metric')
MODEL_LOSS = Gauge('model_loss', 'Model loss metric')
TRAINED_MODELS = Counter('trained_models_total', 'Total number of trained models')

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'recommendation-models')

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
PROCESSED_DATA_TOPIC = os.getenv('PROCESSED_DATA_TOPIC', 'processed-data')
MODEL_READY_TOPIC = os.getenv('MODEL_READY_TOPIC', 'model-ready')

# Model storage
MODEL_DIR = os.getenv('MODEL_DIR', '/models')

@app.on_event("startup")
def startup_event():
    # Start Prometheus metrics server
    start_http_server(8000)
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    logger.info("Model Training Service started")
    
    # Start background training process
    background_tasks = BackgroundTasks()
    background_tasks.add_task(start_training_pipeline)

def start_training_pipeline():
    """
    Start the model training pipeline
    """
    consumer = KafkaConsumer(
        PROCESSED_DATA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        group_id='model-training-group',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    logger.info(f"Started consuming from topic: {PROCESSED_DATA_TOPIC}")
    
    # Collect data until we have enough for training
    data_buffer = []
    min_training_samples = 1000  # Minimum number of samples to start training
    
    for message in consumer:
        try:
            # Extract data from message
            batch_data = message.value
            data_buffer.extend(batch_data)
            
            logger.info(f"Received batch of {len(batch_data)} records. Total buffer size: {len(data_buffer)}")
            
            # Check if we have enough data to train
            if len(data_buffer) >= min_training_samples:
                # Convert to DataFrame
                df = pd.DataFrame(data_buffer)
                
                # Train models
                train_models(df)
                
                # Clear buffer after training
                data_buffer = []
                
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

@TRAINING_DURATION.time()
def train_models(df):
    """
    Train recommendation models
    
    Args:
        df: DataFrame with processed data
    """
    logger.info(f"Starting model training with {len(df)} samples")
    
    try:
        with mlflow.start_run(run_name=f"training-run-{int(time.time())}"):
            # Log dataset info
            mlflow.log_param("dataset_size", len(df))
            mlflow.log_param("n_users", df['user_id'].nunique())
            mlflow.log_param("n_items", df['movie_id'].nunique())
            
            # 1. Apply enhanced feature engineering
            feature_engineering = EnhancedFeatureEngineering(
                user_col='user_id',
                item_col='movie_id',
                rating_col='rating',
                timestamp_col='timestamp' if 'timestamp' in df.columns else None
            )
            
            # Apply feature engineering
            enhanced_df = feature_engineering.fit(df).transform(df)
            
            # Log feature info
            mlflow.log_param("n_features", enhanced_df.shape[1])
            
            # 2. Train Neural Collaborative Filtering model
            ncf_model, user_encoder, item_encoder, history = train_ncf_model(
                df[['user_id', 'movie_id', 'rating']],
                embedding_size=32,
                layers=[64, 32, 16],
                epochs=5,
                batch_size=256
            )
            
            # Log NCF metrics
            ncf_val_loss = history.history['val_loss'][-1]
            ncf_val_accuracy = history.history['val_accuracy'][-1]
            
            mlflow.log_metric("ncf_val_loss", ncf_val_loss)
            mlflow.log_metric("ncf_val_accuracy", ncf_val_accuracy)
            
            # Update Prometheus metrics
            MODEL_LOSS.set(ncf_val_loss)
            MODEL_ACCURACY.set(ncf_val_accuracy)
            
            # Save NCF model
            ncf_model_path = os.path.join(MODEL_DIR, 'ncf_model')
            ncf_model.save_model(ncf_model_path)
            
            # Log model to MLflow
            mlflow.log_artifact(ncf_model_path)
            import mlflow
            mlflow.log_model(model, "recsys-model")
            
            # 3. Train GNN model (simplified for demonstration)
            # In a real implementation, we would train the GNN model here
            
            # Increment trained models counter
            TRAINED_MODELS.inc()
            
            logger.info(f"Model training completed successfully. Models saved to {MODEL_DIR}")
            
            # Notify that model is ready
            notify_model_ready({
                "model_type": "ncf",
                "model_path": ncf_model_path,
                "metrics": {
                    "val_loss": ncf_val_loss,
                    "val_accuracy": ncf_val_accuracy
                }
            })
            
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def notify_model_ready(model_info):
    """
    Notify that a model is ready for deployment
    
    Args:
        model_info: Dictionary with model information
    """
    # In a real implementation, we would send a message to Kafka
    # to notify the deployment service that a new model is ready
    logger.info(f"Model ready for deployment: {model_info}")

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@app.get("/metrics")
def get_metrics():
    """
    Get current metrics
    """
    return {
        "trained_models": TRAINED_MODELS._value.get(),
        "model_accuracy": MODEL_ACCURACY._value,
        "model_loss": MODEL_LOSS._value
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)