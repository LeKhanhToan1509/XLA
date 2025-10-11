import cv2
import numpy as np
import os
import random
from tqdm import tqdm

IMG_SIZE = 64
DATA_DIR = "data_best"
CLASSES = ["circle", "square", "triangle"]
NUM_IMAGES = 1000

# --- Utility noise + transform functions ---
def add_gaussian_noise(img, var=40):
    sigma = var ** 0.5
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise
    return np.clip(img, 0, 255).astype(np.uint8)

def add_salt_pepper_noise(img, amount=0.02):
    noisy = img.copy()
    num_salt = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 255

    num_pepper = int(amount * img.size * 0.5)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
    noisy[coords[0], coords[1], :] = 0
    return noisy

def random_background(size):
    base = np.random.randint(30, 100, (size, size, 3), dtype=np.uint8)
    overlay = cv2.GaussianBlur(base, (3,3), random.uniform(0,1.5))
    return overlay

# --- Core shape generator ---
def create_shape(shape_type, size=IMG_SIZE):
    img = random_background(size)

    # Random vị trí, kích thước, góc xoay
    center = (random.randint(15, size-15), random.randint(15, size-15))
    s = random.randint(10, size//3)
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

    # --- Augmentation nhẹ ---
    if random.random() < 0.7:
        img = add_gaussian_noise(img, var=random.randint(20, 60))
    if random.random() < 0.5:
        img = add_salt_pepper_noise(img, amount=random.uniform(0.01, 0.03))
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3,3), random.uniform(0,1))
    if random.random() < 0.4:
        alpha = random.uniform(0.8, 1.2)
        img = np.clip(img * alpha, 0, 255).astype(np.uint8)

    return img

# --- Dataset generator ---
def generate_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in CLASSES:
        out_dir = os.path.join(DATA_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)
        print(f"⚙️ Generating {cls} images...")
        for i in tqdm(range(NUM_IMAGES)):
            img = create_shape(cls)
            cv2.imwrite(os.path.join(out_dir, f"{cls}_{i}.png"), img)

if __name__ == "__main__":
    generate_dataset()
    print("\n✅ Done! Dataset saved in 'data_best/'")
