import os 
import logging
import subprocess
from pathlib import Path
import pandas as pd
import yaml

log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_path = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(file_path,mode='a')
file_handler.setLevel('DEBUG')

formatter  = logging.Formatter(
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "kaggle_room_street_data"
HOUSE_DATA_PATH = RAW_DATA_DIR / "house_data"
STREET_DATA_PATH = RAW_DATA_DIR / "street_data"

def pull_data_from_remote():
    """
    Executes 'dvc pull' to download the latest versioned data from AWS S3
    and restore it to the local data directory.
    """
    logger.debug("Starting DVC pull to ensure data is in local cache...")
    try:
        # Pulls all DVC-tracked files (house_data, street_data)
        # This is the robust ingestion step: fetching from S3.
        result = subprocess.run(["dvc", "pull"], check=True, capture_output=True, text=True)
        logger.info("DVC Pull Successful.")
        logger.debug(f"DVC Pull Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error during DVC pull: {e}")
        logger.debug(f"Stdout: {e.stdout}")
        logger.debug(f"Stderr: {e.stderr}")
        raise RuntimeError("Failed to ingest data from DVC remote.") from e

def create_image_manifest(data_path: Path) -> list[Path]:
    """
    Scans a local directory for image files (jpg, png) and returns a list of their local paths.
    """
    logger.debug(f"Creating image manifest for directory: {data_path}...")
    # Check if the directory exists and contains data (i.e., dvc pull worked)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}. DVC pull may have failed.")
    
    # Use glob to find all common image files recursively
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        # We use data_path.rglob(ext) to search all subfolders as well
        image_paths.extend(data_path.rglob(ext))

    if not image_paths:
        raise ValueError(f"No image files found in the directory: {data_path}. Check the file extensions.")
    logger.info(f"Found {len(image_paths)} images in {data_path}.")
    return image_paths

def load_local_data_paths() -> tuple[list[Path], list[Path]]:
    """
    Main function to execute the full data ingestion pipeline step.
    1. Ensures data is local via 'dvc pull'.
    2. Creates lists of local file paths for 'house' and 'street' data.
    """
    # 1. Ensure the data is present locally (the actual ingestion/download from S3)
    pull_data_from_remote()

    logger.debug("Creating image manifests from local directory...")

    # 2. Generate lists of file paths
    house_data_paths = create_image_manifest(HOUSE_DATA_PATH)
    street_data_paths = create_image_manifest(STREET_DATA_PATH)
    
    # Optional: Log/Print summary for robust logging
    logger.info(f"Successfully loaded {len(house_data_paths)} house images.")
    logger.info(f"Successfully loaded {len(street_data_paths)} street images.")

    return house_data_paths, street_data_paths

if __name__ == "__main__":
    # 1. Execute the full ingestion and get the paths
    try:
        logger.info("Starting Data Ingestion Pipeline Stage.")
        house_paths, street_paths = load_local_data_paths()
        
        # 2. Combine the paths and labels into a single DataFrame (Manifest)
        data = {
            'image_path': house_paths + street_paths,
            'label': [0] * len(house_paths) + [1] * len(street_paths) # Assuming 0=house, 1=street
        }
        df = pd.DataFrame(data)
        
        # 3. Save the Manifest to a designated, clean output path
        manifest_path = PROJECT_ROOT / "data" / "processed" / "raw_image_manifest.csv"
        os.makedirs(manifest_path.parent, exist_ok=True)
        df.to_csv(manifest_path, index=False)
        
        logger.info(f"Successfully created and saved image manifest to: {manifest_path}")
        logger.info(f"Total rows in manifest: {len(df)}")
        logger.info("Data Ingestion Pipeline Stage completed successfully.")
        
    except Exception as e:
        logger.error(f"FATAL ERROR in data ingestion: {e}", exc_info=True)
        # Re-raise the exception to stop the overall pipeline
        exit(1)