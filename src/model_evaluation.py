import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix

# --- 1. SETUP & CONFIGURATION ---

# Define Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Artifact Paths
TRAINED_MODEL_PATH = PROJECT_ROOT / "run" / "models" / "trained_model.h5"
TEST_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_manifest.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Logging setup
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_DIR / 'model_evaluation.log', mode='a')
formatter  = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --- 2. EVALUATION FUNCTIONS ---

def load_and_predict(model_path: Path, manifest_path: Path, target_size: tuple):
    """
    Loads the trained model and the test data to generate predictions.
    """
    logger.info("Loading trained model and test data...")
    
    # Check dependencies
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Test manifest not found at: {manifest_path}")

    # Load artifacts
    model = load_model(model_path)
    test_df = pd.read_csv(manifest_path)
    
    # Separate true labels (Y_true)
    y_true = test_df['label'].values
    
    # Keras requires label column to be string for generator to work with the index
    test_df['label'] = test_df['label'].astype(str)
    
    # Data Generator (only rescaling needed for test data)
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=target_size,
        batch_size=32, # Use a fixed batch size for prediction
        class_mode='binary',
        shuffle=False # IMPORTANT: Do not shuffle test data!
    )
    
    # Generate predictions (probabilities)
    logger.info("Generating predictions on the test set...")
    y_pred_probs = model.predict(test_generator)
    
    # Convert probabilities to binary class labels (0 or 1)
    # Since we used sigmoid, threshold is 0.5
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    return y_true, y_pred, y_pred_probs.flatten()


def generate_metrics_report(y_true, y_pred, y_pred_probs) -> dict:
    """
    Calculates key classification metrics and formats them for JSON logging.
    """
    logger.info("Calculating final metrics...")
    
    # 1. Core Metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_probs)
    
    # 2. Classification Report (Per Class Metrics)
    report = classification_report(y_true, y_pred, target_names=['house (0)', 'street (1)'], output_dict=True)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    # Simplify the classification report into a clean JSON structure
    summary = {
        "accuracy": float(accuracy),
        "auc_score": float(auc_score),
        "confusion_matrix": cm,
        "classification_report": {
            "house": report['house (0)'],
            "street": report['street (1)'],
            "macro_avg": report['macro avg'],
            "weighted_avg": report['weighted avg']
        }
    }
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    logger.info(f"Model AUC Score: {auc_score:.4f}")
    
    return summary


# --- 3. MAIN EXECUTION ---

def main():
    logger.info("--- Starting Model Evaluation Stage ---")
    
    # Hardcode the target size based on model_building params (100, 100)
    TARGET_SIZE = (100, 100)

    # 1. Predict on Test Data
    try:
        y_true, y_pred, y_pred_probs = load_and_predict(
            TRAINED_MODEL_PATH, 
            TEST_MANIFEST_PATH, 
            TARGET_SIZE
        )
    except FileNotFoundError as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error("Please ensure model_training.py ran successfully to create the trained model.")
        exit(1)
    
    # 2. Generate Metrics
    metrics_report = generate_metrics_report(y_true, y_pred, y_pred_probs)
    
    # 3. Save Metrics to JSON file (Output Artifact)
    try:
        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics_report, f, indent=4)
        
        logger.info(f"Successfully saved metrics report to: {METRICS_PATH}")
    except Exception as e:
        logger.error(f"Failed to save metrics JSON: {e}")
        exit(1)
        
    logger.info("--- Model Evaluation Stage Complete ---")


if __name__ == "__main__":
    # To run this script, you may need to read input_shape from params.yaml dynamically
    # For a simplified run, we use the hardcoded 100x100 size established in the build stage.
    main()