import pandas as pd
import numpy as np
import os
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

# --- 1. SETUP & CONFIGURATION ---

# Define Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "raw_image_manifest.csv"
PARAMS_FILE = PROJECT_ROOT / "params.yaml"
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Logging setup (Reusing the robust setup from data_ingestion)
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_DIR / 'data_preprocessing.log', mode='a')
formatter  = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load Parameters
try:
    with open(PARAMS_FILE, 'r') as f:
        params = yaml.safe_load(f)['data_preprocessing']
except Exception as e:
    logger.error(f"Failed to load parameters from {PARAMS_FILE}: {e}")
    exit(1)


# --- 2. IMAGE AUGMENTATION LOGIC ---

def augment_and_save_images(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Performs augmentation (rotation, flipping) on existing images, saves the new 
    augmented images in place (using new filenames), and updates the DataFrame 
    with the paths to the new images. This increases the total dataset size.
    """
    augmented_data = []
    
    rotations = params.get('augment_rotations', [])
    do_flips = params.get('augment_flips', False)

    logger.info(f"Starting augmentation: Rotations={rotations}, Flips={do_flips}")

    for index, row in df.iterrows():
        original_path = Path(row['image_path'])
        original_label = row['label']
        
        # 1. Keep the original image path
        augmented_data.append({'image_path': str(original_path), 'label': original_label})

        # Ensure the output directory exists
        output_dir = original_path.parent
        
        # 2. Augment by Rotation
        for angle in rotations:
            try:
                img = Image.open(original_path)
                
                # Rotate the image
                augmented_img = img.rotate(angle, expand=True)
                
                # Create a new unique filename
                new_filename = f"{original_path.stem}_rot{angle}{original_path.suffix}"
                new_path = output_dir / new_filename
                
                # Save the augmented image in the SAME location
                augmented_img.save(new_path)
                
                augmented_data.append({'image_path': str(new_path), 'label': original_label})
                img.close()
            except Exception as e:
                logger.warning(f"Failed to rotate image {original_path.name} by {angle} degrees: {e}")

        # 3. Augment by Flipping
        if do_flips:
            try:
                img = Image.open(original_path)
                
                # Horizontal Flip
                h_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                h_flip_path = output_dir / f"{original_path.stem}_hf{original_path.suffix}"
                h_flip.save(h_flip_path)
                augmented_data.append({'image_path': str(h_flip_path), 'label': original_label})
                
                # Vertical Flip
                v_flip = img.transpose(Image.FLIP_TOP_BOTTOM)
                v_flip_path = output_dir / f"{original_path.stem}_vf{original_path.suffix}"
                v_flip.save(v_flip_path)
                augmented_data.append({'image_path': str(v_flip_path), 'label': original_label})
                
                img.close()
            except Exception as e:
                logger.warning(f"Failed to flip image {original_path.name}: {e}")
                
    # Create the final, augmented DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    logger.info(f"Augmentation complete. Dataset size increased from {len(df)} to {len(augmented_df)} rows.")
    return augmented_df

# --- 3. DATA SPLITTING LOGIC ---

def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the augmented DataFrame into training and testing sets.
    """
    logger.info(f"Starting data split (Test Size: {test_size}, Random State: {random_state}).")
    
    # Use stratification to ensure class balance in both sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label'] # IMPORTANT for classification tasks
    )
    
    logger.info(f"Train split size: {len(train_df)} rows.")
    logger.info(f"Test split size: {len(test_df)} rows.")
    
    return train_df, test_df

# --- 4. MAIN EXECUTION ---

def main():
    logger.info("--- Starting Data Preprocessing Stage ---")
    
    # 1. Load the manifest from the previous stage
    if not RAW_MANIFEST_PATH.exists():
        logger.error(f"Manifest not found: {RAW_MANIFEST_PATH}. Did data_ingestion.py run successfully?")
        exit(1)
        
    df = pd.read_csv(RAW_MANIFEST_PATH)
    logger.info(f"Loaded initial manifest with {len(df)} images.")

    # 2. Perform augmentation (increases dataset size and saves new files)
    augmented_df = augment_and_save_images(df, params)

    # 3. Split the augmented dataset
    test_size = params['test_size']
    random_state = params['random_state']
    train_df, test_df = split_data(augmented_df, test_size, random_state)

    # 4. Save the final training and testing manifests (Output Artifacts)
    train_manifest_path = PROJECT_ROOT / params['train_manifest_out']
    test_manifest_path = PROJECT_ROOT / params['test_manifest_out']
    
    # Ensure output directory exists
    train_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(train_manifest_path, index=False)
    test_df.to_csv(test_manifest_path, index=False)
    
    logger.info(f"Training manifest saved to: {train_manifest_path}")
    logger.info(f"Testing manifest saved to: {test_manifest_path}")
    logger.info("--- Data Preprocessing Pipeline Complete ---")


if __name__ == "__main__":
    main()
