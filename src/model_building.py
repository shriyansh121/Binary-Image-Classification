import os
import yaml
import logging
from pathlib import Path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense 

# --- 1. SETUP & CONFIGURATION ---

# Define Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Define Model Output Path
MODEL_DIR = PROJECT_ROOT / "run" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
UNTRAINED_MODEL_PATH = MODEL_DIR / "untrained_model.h5"

# Logging setup
logger = logging.getLogger('model_building')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_DIR / 'model_building.log', mode='a')
formatter  = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --- 2. MODEL DEFINITION ---

def build_cnn_model(params: dict):
    """
    Builds the specified CNN architecture using parameters from params.yaml.
    """
    p = params['model_building']
    
    logger.info(f"Building CNN model with input shape: {p['input_shape']}")
    
    # Keras Model Definition
    model = Sequential()
    
    # 1. First Convolutional Block
    model.add(Conv2D(p['conv1_filters'], (3, 3), 
                     activation='relu', 
                     input_shape=p['input_shape'],
                     name='conv_1'))
    model.add(MaxPool2D(2, 2, name='pool_1'))
    
    # 2. Second Convolutional Block
    model.add(Conv2D(p['conv2_filters'], (3, 3), 
                     activation='relu',
                     name='conv_2'))
    model.add(MaxPool2D(2, 2, name='pool_2'))
    
    # 3. Flatten and Dense Layers
    model.add(Flatten(name='flatten'))
    
    # First Dense Layer
    model.add(Dense(p['dense1_units'], activation='relu', name='dense_1'))
    
    # Output Layer (Corrected for Binary Classification)
    # The output should be 1 unit with a sigmoid activation for House vs. Street (binary)
    model.add(Dense(p['output_classes'], activation=p['output_activation'], name='output'))
    
    return model


# --- 3. MAIN EXECUTION ---

def main():
    logger.info("--- Starting Model Building Stage ---")
    
    # Load Parameters
    try:
        with open(PARAMS_FILE, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load parameters from {PARAMS_FILE}: {e}")
        exit(1)
        
    # Build the model
    model = build_cnn_model(params)
    
    # Display and log model summary
    model.summary(print_fn=lambda x: logger.info(x))

    # Save the untrained model artifact
    try:
        model.save(UNTRAINED_MODEL_PATH)
        logger.info(f"Successfully built and saved UNTRAINED model to: {UNTRAINED_MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        exit(1)
        
    logger.info("--- Model Building Stage Complete ---")


if __name__ == "__main__":
    main()