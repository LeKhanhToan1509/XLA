"""
Configuration File for ResNet-50 Training
=========================================

Tập trung tất cả hyperparameters và constants quan trọng
để dễ dàng tuning và experiment tracking.

Hyperparameters được chọn dựa trên:
1. Literature review và best practices
2. Hardware constraints (GPU memory)
3. Dataset characteristics
4. Computational budget
"""

# Dataset Configuration
# ====================
NUM_CLASSES = 13  
"""
Số lượng classes trong dataset
- Digits: 0-9 (10 classes)
- Shapes: circle, square, triangle (3 classes)
- Total: 13 classes
- Custom dataset: Adjust accordingly
"""

INPUT_SHAPE = (3, 28, 28)  
"""
Input tensor dimensions (Channels, Height, Width)
- 3: RGB channels (grayscale = 1)
- 24x24: Spatial dimensions (có thể 32x32, 224x224, etc.)
- Nhỏ hơn ImageNet standard (224x224) để fit memory constraints
"""

# Training Configuration  
# ======================
BATCH_SIZE = 16  # Optimized for 4GB VRAM
"""
Số samples mỗi mini-batch
- Trade-off giữa memory usage và gradient stability
- Larger batch: More stable gradients, cần more memory
- Smaller batch: Less memory, more noisy gradients
- Typical values: 16, 32, 64, 128
- Using 4 for 4GB VRAM constraint (can try 8 if stable)
- ⚠️ For 4GB VRAM: BATCH_SIZE=4 is safe, 8 might OOM depending on image size
"""

LEARNING_RATE = 0.001
"""
Adam optimizer learning rate (α)
- Controls step size trong gradient descent
- Too high: Unstable training, oscillation
- Too low: Slow convergence, local minima
- Adam default: 0.001 (good starting point)
- Using 0.0001 for stable training from scratch
"""

EPOCHS = 30
"""
Số epochs để training
- 1 epoch = 1 full pass qua toàn bộ training data
- Monitor validation loss để determine optimal stopping point
- Early stopping nếu overfitting
"""