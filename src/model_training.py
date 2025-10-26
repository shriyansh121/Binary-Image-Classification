import os
import yaml
import logging
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# --- 1. SETUP & CONFIGURATION ---

# Define Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Artifact Paths
UNTRAINED_MODEL_PATH = PROJECT_ROOT / "run" / "models" / "untrained_model.h5"
TRAIN_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "train_manifest.csv"
TEST_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "test_manifest.csv"

# Logging setup (standard MLOps logger)
logger = logging.getLogger('model_training')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_DIR / 'model_training.log', mode='a')
formatter  = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --- 2. MODEL LOADING ---

def load_untrained_model(model_path: Path):
    """
    Loads the saved model architecture from the model_building stage.
    """
    if not model_path.exists():
        logger.error(f"Untrained model not found at: {model_path}")
        raise FileNotFoundError("Model Building artifact missing.")
        
    logger.info(f"Loading untrained model from {model_path}")
    model = load_model(model_path)
    return model


# --- 3. DATA GENERATION AND TRAINING ---

def train_model(model, train_params: dict, data_params: dict):
    """
    Loads data, compiles the model, and performs training.
    """
    logger.info("Starting model training.")
    
    # 3.1. Load Data Manifests
    train_df = pd.read_csv(TRAIN_MANIFEST_PATH)
    test_df = pd.read_csv(TEST_MANIFEST_PATH)
    
    # --- ADDED CODE: Class Count Check ---
    # Assuming label 0 is 'house' and label 1 is 'street'
    train_counts = train_df['label'].value_counts().to_dict()
    test_counts = test_df['label'].value_counts().to_dict()

    logger.info(f"Train set class distribution (0=House, 1=Street): {train_counts}")
    logger.info(f"Test set class distribution (0=House, 1=Street): {test_counts}")
    # ------------------------------------

    # Keras requires the label column to be string for flow_from_dataframe
    train_df['label'] = train_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)
    
    # 3.2. Define Image Generator (Only rescaling is needed here)
    # Note: Augmentation (rotation/flips) was handled by saving new files in preprocess_data.py
    datagen = ImageDataGenerator(rescale=1./255)
    
    # Get image input shape (height, width) for target_size
    H, W, C = data_params['input_shape']
    
    # 3.3. Create Data Generators
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label',
        target_size=(H, W),
        batch_size=train_params['batch_size'],
        class_mode='binary', # Since output layer is 1 unit (sigmoid)
        shuffle=True
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label',
        target_size=(H, W),
        batch_size=train_params['batch_size'],
        class_mode='binary',
        shuffle=False
    )
    
    # 3.4. Compile the Model (Setting up optimizer and loss)
    logger.info(f"Compiling model with optimizer={train_params['optimizer']} and loss={train_params['loss']}")
    
    optimizer = Adam(learning_rate=train_params['learning_rate'])
    model.compile(optimizer=optimizer, 
                  loss=train_params['loss'], 
                  metrics=train_params['metrics'])

    # 3.5. Train the Model
    logger.info(f"Training for {train_params['epochs']} epochs...")
    history = model.fit(
        train_generator,
        epochs=train_params['epochs'],
        validation_data=validation_generator,
        verbose=1
    )
    
    logger.info("Training complete.")
    return model, history


# --- 4. MAIN EXECUTION ---

def main():
    logger.info("--- Starting Model Training Stage ---")
    
    # Load Parameters
    try:
        with open(PARAMS_FILE, 'r') as f:
            full_params = yaml.safe_load(f)
        train_params = full_params['model_training']
        data_params = full_params['model_building']
        trained_model_out_path = PROJECT_ROOT / train_params['trained_model_out']
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        exit(1)
        
    # 1. Load Untrained Model
    model = load_untrained_model(UNTRAINED_MODEL_PATH)
    
    # 2. Train the Model
    try:
        trained_model, history = train_model(model, train_params, data_params)
    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        exit(1)
        
    # 3. Save the Trained Model (Output Artifact)
    try:
        trained_model.save(trained_model_out_path)
        logger.info(f"Successfully saved TRAINED model to: {trained_model_out_path}")
    except Exception as e:
        logger.error(f"Failed to save trained model: {e}")
        exit(1)
        
    logger.info("--- Model Training Stage Complete ---")


if __name__ == "__main__":
    main()
