# ai/data/dataloader.py
"""
Load dữ liệu từ thư mục train/val/test (chữ số 0-9 + circle/square/triangle).
Hỗ trợ one-hot encoding, caching và numpy file để tăng tốc.
"""

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from ai.configs.config import ROOT_DIR, NUM_CLASSES, INPUT_SHAPE

# Cache để tránh load lại
_CACHE = {}

def get_cache_path(split):
    """Get path for cached numpy file"""
    cache_dir = os.path.join(ROOT_DIR, '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f'{split}_data.npz')

class DataLoader:
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Sử dụng đúng thư viện (NumPy hoặc CuPy)
        if hasattr(X, '__cuda_array_interface__'):  # CuPy array
            try:
                import cupy as cp
                self.xp = cp
            except:
                self.xp = np
        else:
            self.xp = np
        
        self.indices = self.xp.arange(len(X))
        self.encoder = OneHotEncoder(sparse_output=False)
        self.reset()

    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            self.xp.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= len(self.X):
            raise StopIteration
        end_idx = min(self.current_idx + self.batch_size, len(self.X))
        batch_indices = self.indices[self.current_idx:end_idx]
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]
        self.current_idx = end_idx
        return X_batch, y_batch

    def one_hot(self, y, num_classes=NUM_CLASSES):
        """Convert labels to one-hot encoding với đúng số classes"""
        if len(y.shape) > 1 and y.shape[1] == num_classes:
            return y
        
        # Tạo one-hot matrix manually để đảm bảo đúng số classes
        batch_size = len(y)
        
        # Check if y is cupy array
        if hasattr(y, 'get'):  # CuPy array
            y_cpu = y.get()
            use_cupy = True
        else:
            y_cpu = y
            use_cupy = False
        
        # Tạo one-hot với NumPy trước
        one_hot_matrix = np.zeros((batch_size, num_classes), dtype=np.float32)
        for i, label in enumerate(y_cpu):
            if 0 <= label < num_classes:
                one_hot_matrix[i, int(label)] = 1.0
        
        # Chuyển sang CuPy nếu input là CuPy
        if use_cupy:
            return self.xp.asarray(one_hot_matrix)
        
        return one_hot_matrix

def load_dataset(split='train', use_cache=True, use_disk_cache=True):
    """Load dataset với tối ưu tốc độ - batch processing + memory cache + disk cache"""
    
    # Kiểm tra memory cache trước
    if use_cache and split in _CACHE:
        print(f"⚡ Using memory cached {split} data")
        return _CACHE[split]
    
    # Kiểm tra disk cache
    cache_path = get_cache_path(split)
    if use_disk_cache and os.path.exists(cache_path):
        try:
            print(f"⚡ Loading from disk cache: {cache_path}")
            data = np.load(cache_path)
            X, y = data['X'], data['y']
            print(f"✅ Loaded {split} from cache: {len(X)} samples, shape {X.shape}")
            if use_cache:
                _CACHE[split] = (X, y)
            return X, y
        except Exception as e:
            print(f"⚠️  Cache load failed: {e}, loading from images...")
    
    class_to_idx = {str(i): i for i in range(10)}  # 0-9
    class_to_idx.update({'circle': 10, 'square': 11, 'triangle': 12})
    
    split_dir = os.path.join(ROOT_DIR, split)
    
    # Thu thập tất cả file paths trước
    all_files = []
    all_labels = []
    
    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = os.path.join(split_dir, str(cls_name))
        if not os.path.exists(cls_dir):
            print(f"⚠️  Directory {cls_dir} not found!")
            continue
        img_files = glob.glob(os.path.join(cls_dir, "*.png")) + \
                    glob.glob(os.path.join(cls_dir, "*.jpg"))
        all_files.extend(img_files)
        all_labels.extend([cls_idx] * len(img_files))
    
    if len(all_files) == 0:
        raise ValueError(f"No images in {split_dir}!")
    
    print(f"🔄 Loading {len(all_files)} images from {split}...")
    
    # Pre-allocate array để tăng tốc
    X = np.zeros((len(all_files), INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Load với progress bar duy nhất
    valid_count = 0
    for idx, img_path in enumerate(tqdm(all_files, desc=f"Loading {split}")):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize
        img = cv2.resize(img, (INPUT_SHAPE[1], INPUT_SHAPE[2]))
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize và transpose
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        
        X[valid_count] = img
        valid_count += 1
    
    # Trim invalid images
    if valid_count < len(all_files):
        X = X[:valid_count]
        y = y[:valid_count]
    
    print(f"✅ Loaded {split}: {len(X)} samples, shape {X.shape}")
    
    # Save to disk cache
    if use_disk_cache:
        try:
            print(f"💾 Saving to disk cache: {cache_path}")
            np.savez_compressed(cache_path, X=X, y=y)
            print(f"✅ Cache saved successfully")
        except Exception as e:
            print(f"⚠️  Cache save failed: {e}")
    
    # Cache in memory
    if use_cache:
        _CACHE[split] = (X, y)
    
    return X, y