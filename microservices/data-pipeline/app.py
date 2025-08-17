import os
import time
import json
import logging
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from kafka import KafkaProducer, KafkaConsumer
from prometheus_client import start_http_server, Counter, Gauge

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Data Pipeline Service", version="1.0.0")

# Prometheus metrics
PROCESSED_RECORDS = Counter('processed_records_total', 'Total number of processed records')
PROCESSING_TIME = Gauge('processing_time_seconds', 'Time taken to process a batch of records')

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
RAW_DATA_TOPIC = os.getenv('RAW_DATA_TOPIC', 'raw-data')
PROCESSED_DATA_TOPIC = os.getenv('PROCESSED_DATA_TOPIC', 'processed-data')

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

@app.on_event("startup")
def startup_event():
    # Start Prometheus metrics server
    start_http_server(8000)
    logger.info("Data Pipeline Service started")
    
    # Start background data processing
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_data_stream)

def process_data_stream():
    """
    Continuously process data from Kafka stream
    """
    consumer = KafkaConsumer(
        RAW_DATA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    logger.info(f"Started consuming from topic: {RAW_DATA_TOPIC}")
    
    for message in consumer:
        try:
            start_time = time.time()
            
            # Extract data from message
            data = message.value
            logger.info(f"Processing batch of {len(data)} records")
            
            # Convert to DataFrame for processing
            df = pd.DataFrame(data)
            
            # Process the data
            processed_df = process_data(df)
            
            # Convert back to records
            processed_records = processed_df.to_dict('records')
            
            # Send to processed data topic
            producer.send(PROCESSED_DATA_TOPIC, processed_records)
            
            # Update metrics
            PROCESSED_RECORDS.inc(len(processed_records))
            processing_time = time.time() - start_time
            PROCESSING_TIME.set(processing_time)
            
            logger.info(f"Processed and sent {len(processed_records)} records in {processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

def process_data(df):
    """
    Process the raw data
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Processed DataFrame
    """
    # Implement data processing logic here
    # For example:
    
    # 1. Handle missing values
    df = df.fillna({
        'rating': df['rating'].mean() if 'rating' in df.columns else 0,
        'timestamp': pd.Timestamp.now().timestamp()
    })
    
    # 2. Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # 3. Normalize ratings
    if 'rating' in df.columns:
        df['normalized_rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
    
    return df

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
        "processed_records": PROCESSED_RECORDS._value.get(),
        "average_processing_time": PROCESSING_TIME._value
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)