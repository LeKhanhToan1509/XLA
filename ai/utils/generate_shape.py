try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️  Using CPU (NumPy)")
    GPU_AVAILABLE = False
import cv2
import os
import random
from tqdm import tqdm

# --- Config ---
IMG_SIZE = 28
CLASSES = ["circle", "square", "triangle"]
NUM_IMAGES = 1000  # tổng mỗi class
ROOT_DIR = r"D:\PTIT\XLA\ai\data"
SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

# --- Utility functions ---
def add_gaussian_noise(img, var=20):
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, amount=0.01):
    noisy = img.copy()
    num_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    num_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def random_background(size):
    base = np.random.randint(40, 120, (size, size, 3), dtype=np.uint8)
    overlay = cv2.GaussianBlur(base, (3,3), random.uniform(0,1.2))
    return overlay

# --- Core shape generator ---
def create_shape(shape_type, size=IMG_SIZE):
    img = random_background(size)

    center = (random.randint(6, size-6), random.randint(6, size-6))
    s = random.randint(4, size//3)
    angle = random.randint(0, 180)
    color = (255, 255, 255)

    if shape_type == "circle":
        cv2.circle(img, center, s, color, -1)

    elif shape_type == "square":
        rect = np.array([
            [center[0]-s, center[1]-s],
            [center[0]+s, center[1]-s],
            [center[0]+s, center[1]+s],
            [center[0]-s, center[1]+s]
        ], np.int32)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rect = np.int32(cv2.transform(np.array([rect]), M))
        cv2.fillPoly(img, [rect], color)

    elif shape_type == "triangle":
        pts = np.array([
            [center[0], center[1]-s],
            [center[0]-s, center[1]+s],
            [center[0]+s, center[1]+s]
        ], np.int32)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        pts = np.int32(cv2.transform(np.array([pts]), M))
        cv2.fillPoly(img, [pts], color)

    # Augmentation
    if random.random() < 0.7:
        img = add_gaussian_noise(img, var=random.randint(10, 40))
    if random.random() < 0.4:
        img = add_salt_pepper_noise(img, amount=random.uniform(0.005, 0.02))
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3,3), random.uniform(0,1))
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)

    return img

# --- Dataset generator ---
def generate_dataset():
    # Tạo folder train/val/test + class
    for split in SPLIT_RATIO.keys():
        for cls in CLASSES:
            os.makedirs(os.path.join(ROOT_DIR, split, cls), exist_ok=True)

    for cls in CLASSES:
        print(f"⚙️ Generating {cls}...")
        for i in tqdm(range(NUM_IMAGES)):
            img = create_shape(cls)
            
            # xác định subset (train/val/test)
            r = random.random()
            if r < SPLIT_RATIO["train"]:
                subset = "train"
            elif r < SPLIT_RATIO["train"] + SPLIT_RATIO["val"]:
                subset = "val"
            else:
                subset = "test"
            
            out_path = os.path.join(ROOT_DIR, subset, cls, f"{cls}_{i}.png")
            cv2.imwrite(out_path, img)

    print("\n✅ Done! Dataset saved to:")
    for k in SPLIT_RATIO.keys():
        print("   ", os.path.join(ROOT_DIR, k))

# --- Run ---
if __name__ == "__main__":
    generate_dataset()
