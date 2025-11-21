
# Configuration for paths and parameters
DATA_DIR = "data/DeepPCB"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_CLASSES = 6  # adjust based on dataset labels
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
