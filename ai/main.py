"""
Main Training Script for ResNet50
==================================

Script này load data, train ResNet50 model, và evaluate performance.

Workflow:
1. Load và preprocess data từ thư mục train/val/test
2. Initialize ResNet50 model
3. Training loop với Adam optimizer
4. Validation sau mỗi epoch
5. Save model checkpoints
6. Final evaluation trên test set
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️ Using CPU (NumPy)")
    GPU_AVAILABLE = False

import cv2
import os
from tqdm import tqdm
import pickle
from model.resnet import ResNet50
from data.dataloader import DataLoader
from configs.config import BATCH_SIZE, EPOCHS, INPUT_SHAPE, NUM_CLASSES
from training.train import train_model, evaluate

def load_images_from_folder(root_folder, img_size=(28, 28)):
    """
    Load images từ folder structure: root/class_name/*.png
    
    Đầu vào:
    - root_folder: Path đến thư mục chứa các class folders
    - img_size: Target size để resize images
    
    Đầu ra:
    - X: Array of images (N, C, H, W) - normalized [0,1]
    - y: Array of labels (N,)
    - class_names: List tên các classes
    """
    class_names = sorted(os.listdir(root_folder))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    images = []
    labels = []
    
    print(f"\n📂 Loading data from: {root_folder}")
    print(f"🏷️  Classes found: {class_names}")
    
    for class_name in tqdm(class_names, desc="Loading classes"):
        class_folder = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
            
        class_idx = class_to_idx[class_name]
        
        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Resize
            img = cv2.resize(img, img_size)
            
            # Convert BGR to RGB và normalize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # Transpose to (C, H, W) format
            img = img.transpose(2, 0, 1)
            
            images.append(img)
            labels.append(class_idx)
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"✅ Loaded {len(X)} images")
    print(f"📊 Data shape: {X.shape}")
    print(f"🎯 Labels shape: {y.shape}")
    print(f"📈 Classes distribution:")
    for idx, name in enumerate(class_names):
        count = np.sum(y == idx)
        print(f"   {name}: {count} samples")
    
    return X, y, class_names


def save_model(model, filepath):
    """Save model weights"""
    print(f"\n💾 Saving model to {filepath}")
    with open(filepath, 'wb') as f:
        pickle.dump({
            'params': model.params,
            'moving_means': {id(layer): layer.moving_mean for layer in model.all_layers if hasattr(layer, 'moving_mean')},
            'moving_vars': {id(layer): layer.moving_var for layer in model.all_layers if hasattr(layer, 'moving_var')}
        }, f)
    print("✅ Model saved successfully!")


def load_model(model, filepath):
    """Load model weights"""
    print(f"\n📥 Loading model from {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        model.params = data['params']
        # Restore BatchNorm moving averages
        for layer in model.all_layers:
            if hasattr(layer, 'moving_mean'):
                layer_id = id(layer)
                if layer_id in data['moving_means']:
                    layer.moving_mean = data['moving_means'][layer_id]
                    layer.moving_var = data['moving_vars'][layer_id]
    print("✅ Model loaded successfully!")


def main():
    """
    Main training pipeline
    """
    print("=" * 60)
    print("🚀 ResNet50 Training Pipeline")
    print("=" * 60)
    
    # Paths
    DATA_ROOT = r"d:\PTIT\XLA\ai\data"
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    VAL_DIR = os.path.join(DATA_ROOT, "val")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    MODEL_SAVE_PATH = "resnet50_model.pkl"
    
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: Training directory not found: {TRAIN_DIR}")
        return
    
    # Load data
    print("\n" + "=" * 60)
    print("📊 STEP 1: LOADING DATA")
    print("=" * 60)
    
    # Lấy img_size từ config
    img_h, img_w = INPUT_SHAPE[1], INPUT_SHAPE[2]
    
    X_train, y_train, class_names = load_images_from_folder(TRAIN_DIR, img_size=(img_h, img_w))
    X_val, y_val, _ = load_images_from_folder(VAL_DIR, img_size=(img_h, img_w))
    X_test, y_test, _ = load_images_from_folder(TEST_DIR, img_size=(img_h, img_w))
    
    num_classes = len(class_names)
    print(f"\n🎯 Number of classes: {num_classes}")
    
    # Update config nếu khác
    if num_classes != NUM_CLASSES:
        print(f"⚠️  Warning: Config NUM_CLASSES ({NUM_CLASSES}) != actual classes ({num_classes})")
        print(f"   Using actual number: {num_classes}")
    
    # Training
    print("\n" + "=" * 60)
    print("🏋️  STEP 2: TRAINING MODEL")
    print("=" * 60)
    
    model = train_model(X_train, y_train, X_val, y_val, num_classes=num_classes)
    
    # Save model
    save_model(model, MODEL_SAVE_PATH)
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("🧪 STEP 3: FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    test_loader = DataLoader(X_test, y_test, BATCH_SIZE, shuffle=False)
    test_acc = evaluate(model, test_loader, num_classes)
    
    print(f"\n🎯 Final Test Accuracy: {test_acc * 100:.2f}%")
    
    # Per-class accuracy
    print("\n📊 Per-class accuracy:")
    model.set_inference(True)
    for idx, class_name in enumerate(class_names):
        class_mask = y_test == idx
        if np.sum(class_mask) == 0:
            continue
        X_class = X_test[class_mask]
        y_class = y_test[class_mask]
        y_pred = model.predict(X_class)
        acc = np.mean(y_pred == y_class)
        print(f"   {class_name:12s}: {acc * 100:5.2f}% ({np.sum(y_pred == y_class)}/{len(y_class)})")
    model.set_inference(False)
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
