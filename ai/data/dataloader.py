"""
Data Loading Module
==================

DataLoader cho việc training Neural Networks:
- Chia data thành các batches
- Shuffle data mỗi epoch để tránh overfitting  
- One-hot encoding cho labels
- Iterator pattern để duyệt qua data

Tại sao cần batching:
1. Memory efficiency: Không thể load toàn bộ dataset vào RAM
2. Gradient stability: Mini-batch gradient descent ổn định hơn SGD
3. Parallelization: GPU xử lý batch hiệu quả hơn single sample

Đầu vào:
- X: Data features (N, channels, height, width) cho images
- y: Labels (N,) - integer class indices
- batch_size: Số samples trong mỗi batch
- shuffle: Boolean, có shuffle data không

Đầu ra (mỗi iteration):
- batch_X: Features của batch (batch_size, channels, height, width)
- batch_y: Labels của batch (batch_size,)
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️ Using CPU (NumPy)")
    GPU_AVAILABLE = False

from configs.config import BATCH_SIZE

class DataLoader:
    def __init__(self, X, y, batch_size=BATCH_SIZE, shuffle=True):
        """
        Khởi tạo DataLoader
        
        Đầu vào:
        - X: Feature data với shape (N, C, H, W)
              N: Số samples
              C: Số channels (3 cho RGB, 1 cho grayscale)  
              H, W: Height và Width của images
        - y: Labels với shape (N,) - integer class indices từ 0 đến num_classes-1
        - batch_size: Kích thước mỗi batch
        - shuffle: Có shuffle data order mỗi epoch không
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))  # Tạo array indices để shuffle
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.X):
            raise StopIteration
        end = min(self.current + self.batch_size, len(self.X))
        batch_indices = self.indices[self.current:end]
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]
        self.current = end
        return batch_X, batch_y

    def one_hot(self, y, num_classes):
        """
        One-Hot Encoding cho labels
        
        Công thức One-Hot Encoding:
        Nếu y[i] = k, thì one_hot[i] = [0,0,...,1,...,0] với 1 ở vị trí k
        
        Ví dụ: 
        - y = [0, 2, 1] với num_classes = 3
        - one_hot = [[1,0,0], [0,0,1], [0,1,0]]
        
        Đầu vào:
        - y: Class indices (batch_size,) với values từ 0 đến num_classes-1
        - num_classes: Tổng số classes
        
        Đầu ra:
        - Z: One-hot encoded matrix (batch_size, num_classes)
        """
        N = y.shape[0]
        Z = np.zeros((N, num_classes))
        Z[np.arange(N), y] = 1  # Set 1 tại position tương ứng với class
        return Z
