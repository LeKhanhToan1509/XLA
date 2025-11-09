# ai/configs/config.py
import os
from dotenv import load_dotenv

load_dotenv()

NUM_CLASSES = int(os.getenv('NUM_CLASSES', 13))  # 0-9 (10) + 3 shapes = 13
INPUT_SHAPE = (3, 28, 28)  # (C, H, W) - RGB
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))  # Tăng lên 128 để GPU hiệu quả hơn
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
EPOCHS = 30
ROOT_DIR = os.getenv('ROOT_DIR', './data')
DEVICE = 'cuda' if os.getenv('GPU_AVAILABLE', 'False').lower() == 'true' else 'cpu'