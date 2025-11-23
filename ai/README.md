# LeNet-5 Implementation - MNIST + Shapes Classification

## 📋 Tổng quan

Implement **LeNet-5 CNN từ đầu** (from scratch) với **NumPy thuần** (không dùng TensorFlow, PyTorch hay bất kỳ deep learning framework nào) để phân loại 13 classes:
- **10 classes digits**: Chữ số 0-9 từ MNIST dataset
- **3 classes shapes**: Hình tròn (circle), vuông (square), tam giác (triangle) - tự generate

### ✨ Đặc điểm nổi bật

- ✅ **100% From-scratch**: Chỉ dùng NumPy, implement tất cả operations (conv, pooling, backprop)
- ✅ **SDLM Optimizer**: Stochastic Diagonal Levenberg-Marquardt cho adaptive learning rate
- ✅ **RBF Output Layer**: Radial Basis Function với bitmap patterns (7×12 ASCII-style)
- ✅ **C3 Layer Mapping**: Theo paper gốc của LeCun (không phải full connection)
- ✅ **Flexible Architecture**: Hỗ trợ 10 classes (MNIST) hoặc 13 classes (MNIST+shapes)
- ✅ **Model Checkpointing**: Auto-save weights mỗi epoch để tiếp tục training hoặc analyze

## 📁 Cấu trúc thư mục

```
ai/
├── main.py                 # Entry point - chạy toàn bộ pipeline
├── train_lenet.py          # Training module với đầy đủ functions
├── README.md               # File này
├── requirements.txt        # Dependencies (numpy, opencv-python, pillow, pickle)
├── models/                 # Model checkpoints (auto-created)
│   ├── model_weights_0.pkl     # Checkpoint epoch 0
│   ├── model_weights_1.pkl     # Checkpoint epoch 1
│   ├── ...
│   └── model_weights_final.pkl # Best model
└── utils/                  # Core implementation
    ├── __init__.py
    ├── LayerObjects.py         # LeNet-5 class + All layers
    ├── Convolution_util.py     # Conv forward/backward/SDLM
    ├── Pooling_util.py         # Pooling forward/backward
    ├── Activation_util.py      # LeNet5_squash activation
    ├── RBF_initial_weight.py   # Bitmap patterns (10 digits + 3 shapes)
    ├── utils_func.py           # Data loading, normalization, mini-batch
    └── generate_shape.py       # Generate synthetic shape images

data/                       # Dataset (không trong repo)
├── train/
│   ├── 0/, 1/, ..., 9/        # MNIST digits
│   ├── circle/                # Generated circles
│   ├── square/                # Generated squares
│   └── triangle/              # Generated triangles
├── test/
│   └── (same structure)
└── val/
    └── (same structure)
```

## 🚀 Hướng dẫn sử dụng

### Bước 0: Chuẩn bị môi trường

**Yêu cầu**: Python 3.6+ (tested với Python 3.6 trên Windows 10)

```bash
# Cài đặt dependencies
pip install numpy opencv-python pillow

# Hoặc dùng requirements.txt
pip install -r ai/requirements.txt
```

### Bước 1: Chuẩn bị dataset

#### Option A: Dùng MNIST digits only (10 classes)

Download MNIST từ [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/) hoặc dùng `MNIST_auto_Download.py`.

Cấu trúc thư mục:
```
data/
├── train/
│   ├── 0/ (ảnh số 0)
│   ├── 1/ (ảnh số 1)
│   └── ...
└── test/
    └── (same structure)
```

#### Option B: MNIST + Shapes (13 classes) - **Recommended**

**1.1. Generate shape dataset**

```bash
cd e:\PTIT\XLA
python ai\utils\generate_shape.py
```

Sẽ tạo:
- **train**: 800 images/class (circle, square, triangle)
- **val**: 100 images/class
- **test**: 100 images/class

Mỗi ảnh:
- Size: 28×28 grayscale
- Format: PNG
- Features: Centered shapes, clean background, high contrast

**1.2. Combine với MNIST**

Đặt MNIST digits vào cùng thư mục `data/train/` và `data/test/`. Kết quả:
```
data/
├── train/
│   ├── 0/, 1/, ..., 9/    ← MNIST digits (10 classes)
│   ├── circle/            ← Generated shapes (3 classes)
│   ├── square/
│   └── triangle/
└── test/
    └── (same structure)
```

### Bước 2: Train model

**Cách 1: Dùng main.py (recommended)**

```bash
cd e:\PTIT\XLA
python ai\main.py
```

**Cách 2: Train trực tiếp**

```bash
cd e:\PTIT\XLA
python ai\train_lenet.py
```

**Cách 3: Dùng Jupyter Notebook**

Mở `LeNet-from-Scratch/LeNet5_train.ipynb` và chạy từng cell.

### Bước 3: Customize hyperparameters

Sửa trong `ai/main.py` hoặc `ai/train_lenet.py`:

```python
model = train_lenet5(
    # Dataset paths
    train_folder=r'E:\PTIT\XLA\data\train',
    test_folder=r'E:\PTIT\XLA\data\test',
    
    # Model config
    n_classes=13,               # 10: MNIST only, 13: MNIST+shapes
    
    # Training config
    num_epochs=20,              # Số epochs (paper gốc dùng 20)
    mini_batch_size=256,        # Batch size (paper không nói rõ, mặc định 256)
    
    # Optimizer config (SDLM)
    lr_global=5e-3,             # Global learning rate (paper: 5e-4, nhưng dùng 5e-3 hội tụ nhanh hơn)
    momentum=0.9,               # Momentum SGD (không có trong paper gốc)
    weight_decay=0,             # L2 regularization (paper gốc không dùng)
    mu=0.01,                    # SDLM diagonal offset parameter
    
    # Checkpoint config
    save_dir='ai/models',       # Thư mục lưu checkpoints
    save_interval=1             # Save mỗi N epochs (1 = save mọi epoch)
)
```

**⚠️ Lưu ý quan trọng:**
- `lr_global`: Paper gốc dùng `5e-4` đến `1e-5`, nhưng implement này cần `5e-3` (×100) do có thể có khác biệt trong SDLM implementation
- `momentum=0.9`: Không có trong paper gốc nhưng giúp hội tụ nhanh hơn
- `n_classes`: **BẮT BUỘC** set đúng số classes trong dataset

## 🏗️ Kiến trúc LeNet-5

```
Input (32×32×1)
    ↓
C1: Conv 5×5, 6 filters → (28×28×6)
    ↓
S2: AvgPool 2×2 → (14×14×6)
    ↓
C3: Conv 5×5, 16 filters (with mapping) → (10×10×16)
    ↓
S4: AvgPool 2×2 → (5×5×16)
    ↓
C5: Conv 5×5, 120 filters → (1×1×120)
    ↓
F6: Fully Connected → (84)
    ↓
RBF: Output layer → (13 classes)
```

## 📊 Features

- ✅ **From-scratch implementation**: NumPy only, không dùng deep learning framework
- ✅ **SDLM optimizer**: Stochastic Diagonal Levenberg-Marquardt cho adaptive learning rate
- ✅ **RBF layer**: Radial Basis Function với bitmap patterns
- ✅ **Data augmentation**: Light noise cho shapes
- ✅ **Model checkpointing**: Lưu weights mỗi epoch
- ✅ **Clean code**: Modular, dễ hiểu và mở rộng

## 📈 Quá trình training

### Console output mẫu

```
======================================================================
                   LeNet-5 Training Pipeline
======================================================================

Starting LeNet-5 training...
======================================================================
LeNet-5 Training - MNIST + Shapes Classification
======================================================================

[1/5] Loading dataset...
Class mapping: {'0': 0, '1': 1, '2': 2, ..., 'circle': 10, 'square': 11, 'triangle': 12}
Loaded 113925 images from E:\PTIT\XLA\data\train
Loaded 14380 images from E:\PTIT\XLA\data\test

✓ Dataset Info:
  - Training samples: 113925
  - Test samples: 14380
  - Image shape: (28, 28)
  - Unique labels: [ 0  1  2  3  4  5  6  7  8  9 10 11 12]
  - Number of classes: 13

[2/5] Preprocessing images...
✓ After padding (pad=2):
  - Training shape: (113925, 32, 32, 1)
  - Test shape: (14380, 32, 32, 1)

[3/5] Initializing LeNet-5 model...
✓ Model architecture:
  C1 (Conv 5x5x1x6) → S2 (AvgPool 2x2) → C3 (Conv 5x5x6x16) →
  S4 (AvgPool 2x2) → C5 (Conv 5x5x16x120) → F6 (FC 120→84) →
  RBF (Output 13 classes)

[4/5] Training model...
Hyperparameters:
  - Epochs: 20
  - Batch size: 256
  - Global LR: 0.005
  - Momentum: 0.9
  - Weight decay: 0
  - SDLM mu: 0.01
----------------------------------------------------------------------
Epoch 1/20 - Loss: 0.5234 - Test Acc: 85.32% - Time: 123.4s
  → Model saved: ai/models/model_weights_0.pkl
Epoch 2/20 - Loss: 0.3421 - Test Acc: 91.45% - Time: 118.2s
  → Model saved: ai/models/model_weights_1.pkl
...
Epoch 20/20 - Loss: 0.0512 - Test Acc: 97.82% - Time: 115.8s
  → Model saved: ai/models/model_weights_19.pkl

[5/5] Final evaluation...

======================================================================
Training Complete!
======================================================================
Final Train Accuracy: 98.91%
Final Test Accuracy: 97.82%

Best Test Accuracy: 98.05% (Epoch 18)

Final model saved: ai/models/model_weights_final.pkl

✅ Training complete! Model checkpoints saved in ai/models/
```

### Thời gian training ước tính

- **1 epoch**: ~2-3 phút (CPU), ~30-60s (GPU - nếu có optimize cho GPU)
- **20 epochs**: ~40-60 phút (CPU)
- Phụ thuộc vào:
  - CPU/GPU
  - Batch size
  - Số lượng training samples

## 🔧 Load và test trained model

### Cách 1: Dùng Python script

```python
import numpy as np
from ai.train_lenet import load_model, evaluate_model
from ai.utils.utils_func import readDatasetFromFolder, zero_pad, normalize

# Load model checkpoint
model = load_model('ai/models/model_weights_19.pkl', n_classes=13)

# Load test data
test_image, test_label = readDatasetFromFolder(r'E:\PTIT\XLA\data\test')
test_image_pad = normalize(zero_pad(test_image[:,:,:,np.newaxis], 2), 'lenet5')

# Evaluate toàn bộ test set
accuracy = evaluate_model(model, test_image_pad, test_label)
print(f"Test Accuracy: {accuracy:.2f}%")

# Predict một ảnh cụ thể
from PIL import Image

img = Image.open('path/to/image.png').convert('L')
img = img.resize((28, 28))
img_array = np.array(img)
img_input = normalize(zero_pad(img_array[np.newaxis, :, :, np.newaxis], 2), 'lenet5')

# Forward pass
_, prediction = model.Forward_Propagation(img_input, np.array([0]), mode='test')
print(f"Predicted class: {prediction[0]}")

# Mapping labels
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'circle', 'square', 'triangle']
print(f"Predicted: {class_names[prediction[0]]}")
```

### Cách 2: Dùng Jupyter Notebook

Mở `LeNet-from-Scratch/a.ipynb` (notebook đã có sẵn để test model):

1. Load model weights từ checkpoint
2. Test với random images từ test set
3. Visualize feature maps của từng layer
4. Compare accuracy across epochs

### Cách 3: Analyze model trong notebook

File `LeNet5_train.ipynb` có đầy đủ visualization:
- Feature maps của C1, S2, C3, S4, C5, F6 layers
- Kernels/filters của conv layers
- RBF layer output
- Training curves (loss, accuracy)

## 📊 Kết quả kỳ vọng

### Accuracy theo paper gốc (MNIST 10 classes)
- **After 1st epoch**: ~93.5%
- **After 20 epochs**: ~98.6%
- **Best reported**: 99.05%

### Accuracy với implementation này (MNIST + Shapes 13 classes)

#### MNIST Digits (0-9):
- **After 1st epoch**: ~88-92%
- **After 20 epochs**: ~96-98%
- Digits thường dễ học hơn shapes do data chất lượng cao

#### Shapes (circle, square, triangle):
- **Với old data generation** (noisy, off-center): ~30-70%
- **Với new data generation** (clean, centered): ~85-95%
- Key: Data quality rất quan trọng cho shapes

#### Overall (13 classes):
- **After 1st epoch**: ~85-90%
- **After 20 epochs**: ~93-97%
- **Best achievable**: ~97-98%

### Training curve patterns
- **Loss**: Giảm nhanh trong 5 epochs đầu, sau đó giảm chậm và ổn định
- **Train accuracy**: Tăng nhanh, có thể đạt >98% sau 10-15 epochs
- **Test accuracy**: Tăng chậm hơn train, gap nhỏ (~1-2%) do model không quá complex

## 📝 Chi tiết kỹ thuật

### Input preprocessing
1. **Original images**: 28×28 grayscale (pixel values 0-255)
2. **Zero-padding**: pad=2 → 32×32 (để C1 output 28×28)
3. **Normalization**: Pixel values → [-0.1, 1.175]
   - Formula: `(pixel / 255) * 1.275 - 0.1`
   - Mean ≈ 0 (theo paper gốc)

### C3 layer mapping (16 feature maps)
Không phải full connection 6→16, mà theo table trong paper:
```
Maps 0-5:   Connect to 3 input maps (6 combinations)
Maps 6-11:  Connect to 4 input maps (6 combinations)
Maps 12-14: Connect to 4 discontinuous maps (3 combinations)
Map 15:     Connect to all 6 input maps (1 full connection)
```

Lý do: Giảm số parameters, tăng diversity của features

### RBF output layer
- **10 digit patterns**: Bitmap 7×12 (84 features) theo ASCII art
- **3 shape patterns**: Bitmap 7×12 tương tự (circle, square, triangle)
- **Loss function**: Euclidean distance giữa F6 output và bitmap pattern
- Không trainable (fixed weights)

### Shapes generation specs
- **Size**: 28×28 pixels
- **Position**: Centered at (14, 14)
- **Circle radius**: 6-10 pixels
- **Square side**: 12-20 pixels (±6-10 from center)
- **Triangle**: Equilateral, vertex pointing up, size 6-10 pixels
- **Background**: Uniform gray (20-80)
- **Foreground**: White-ish (200-255)
- **Augmentation**: Light Gaussian noise (σ=5) on 30% images

### Optimizer: SDLM (Stochastic Diagonal Levenberg-Marquardt)
- **Purpose**: Adaptive learning rate per layer
- **Formula**: `lr = lr_global / (mu + h)`
  - `h`: Approximate diagonal Hessian (từ second derivative)
  - `mu`: Offset parameter (0.01-0.02)
- **Benefits**: Tự động điều chỉnh learning rate dựa trên curvature
- **Paper note**: Implementation này scale lr_global ×100 so với paper gốc

## 🐛 Troubleshooting

### Import errors

**Lỗi**: `ModuleNotFoundError: No module named 'ai'` hoặc `No module named 'utils'`

**Nguyên nhân**: Chạy script từ sai directory

**Giải pháp**:
```bash
# ĐÚNG: Chạy từ root directory
cd e:\PTIT\XLA
python ai\main.py

# SAI: Chạy từ bên trong ai/
cd e:\PTIT\XLA\ai
python main.py  # ← Sẽ lỗi import
```

### Low shapes accuracy

**Triệu chứng**: Shapes accuracy <70% sau 10+ epochs, hoặc model predict tất cả shapes là 1 class

**Nguyên nhân**: 
1. Data generation cũ tạo shapes off-center, noisy, size không consistent
2. RBF bitmap patterns không match với data

**Giải pháp**:
1. **Regenerate data** với `generate_shape.py` mới:
   ```bash
   python ai\utils\generate_shape.py
   ```
2. **Check RBF patterns**: File `ai/utils/RBF_initial_weight.py` phải có `bitmap_circle`, `bitmap_square`, `bitmap_triangle`
3. **Retrain từ đầu**: Xóa old checkpoints và train lại với data mới

### Memory errors

**Lỗi**: `MemoryError` hoặc `numpy.core._exceptions._ArrayMemoryError`

**Nguyên nhân**: Batch size quá lớn cho RAM

**Giải pháp**:
```python
# Giảm batch size
model = train_lenet5(
    mini_batch_size=128,  # Thay vì 256
    # hoặc
    mini_batch_size=64    # Cho RAM <8GB
)
```

### Training quá chậm

**Triệu chứng**: 1 epoch >5 phút

**Nguyên nhân**: 
1. Chạy trên CPU thuần
2. Code NumPy không optimize
3. Dataset quá lớn

**Giải pháp**:
1. **Reduce dataset**: Test với subset nhỏ trước
   ```python
   # Trong train_lenet.py, sau khi load data:
   train_image = train_image[:10000]  # Chỉ lấy 10k samples
   train_label = train_label[:10000]
   ```
2. **Tăng batch size**: Nếu RAM đủ, tăng lên 512
3. **Giảm epochs**: Test với 5-10 epochs trước

### Accuracy không tăng sau nhiều epochs

**Triệu chứng**: Accuracy plateau ở ~70-80%, không cải thiện

**Nguyên nhân**:
1. Learning rate không phù hợp
2. Data imbalance (shapes ít hơn digits rất nhiều)
3. Model đã overfit hoặc underfit

**Giải pháp**:
1. **Adjust learning rate**:
   ```python
   # Tăng lr nếu loss giảm quá chậm
   model = train_lenet5(lr_global=1e-2)  # Default: 5e-3
   
   # Giảm lr nếu loss dao động mạnh
   model = train_lenet5(lr_global=1e-3)
   ```
2. **Check data balance**:
   ```python
   print(np.bincount(train_label))  # Count samples per class
   ```
3. **Visualize predictions**: Dùng notebook `a.ipynb` để xem model predict sai ở đâu

### Model không save được

**Lỗi**: `PicklingError` hoặc `AttributeError` khi save

**Nguyên nhân**: Trying to pickle class definitions thay vì weights

**Giải pháp**: Code đã fix - chỉ save weights dict, không save class object. Nếu vẫn lỗi:
```python
# Trong train_lenet.py, check model_state có đúng format:
model_state = {
    'C1_weight': ConvNet.C1.weight,  # NumPy array
    'C1_bias': ConvNet.C1.bias,      # NumPy array
    # ... (only numpy arrays, no objects)
}
```

### Shapes predict hết là "square"

**Triệu chứng**: Circle và triangle đều được predict là square (label 11)

**Nguyên nhân**: 
1. Data generation tạo shapes quá giống nhau
2. Model collapse do learning rate quá cao hoặc quá thấp
3. RBF patterns không đủ distinctive

**Giải pháp**:
1. **Kiểm tra data**:
   ```python
   import cv2
   import matplotlib.pyplot as plt
   
   # Load và visualize vài ảnh từ mỗi class
   for shape in ['circle', 'square', 'triangle']:
       img = cv2.imread(f'data/train/{shape}/{shape}_00000.png', 0)
       plt.imshow(img, cmap='gray')
       plt.title(shape)
       plt.show()
   ```
2. **Regenerate data** với parameters khác:
   ```python
   # Trong generate_shape.py, tăng diversity:
   radius = random.randint(8, 12)  # Thay vì 6-10
   ```
3. **Check F6 features**: Xem cosine similarity giữa classes trong `a.ipynb`

## 📚 References

### Papers
- **[LeCun et al., 1998]** [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
  - Original LeNet-5 paper
  - Architecture, SDLM optimizer, RBF layer details
  
- **[LeCun et al., 1989]** [Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
  - Early CNN architecture inspiration

### Datasets
- **MNIST**: [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/)
  - 60,000 training + 10,000 test images
  - 28×28 grayscale handwritten digits

### Implementation Notes
- Code structure inspired by [Andrew Ng's Deep Learning Course](https://www.coursera.org/specializations/deep-learning)
- SDLM implementation details from original LeNet-5 paper
- RBF layer bitmap patterns designed theo ASCII art style trong paper

## 🤝 Contributing

Nếu bạn muốn cải thiện code hoặc fix bugs:

1. **Tối ưu performance**: Vectorize operations trong Convolution_util.py
2. **Thêm augmentation**: Random rotation, shift cho shapes
3. **Support GPU**: Thử integrate CuPy thay vì NumPy
4. **Visualization**: Thêm TensorBoard hoặc Weights & Biases logging
5. **More datasets**: Support thêm CIFAR-10, Fashion-MNIST

## 📄 License

Code này dùng cho mục đích học tập và nghiên cứu. Implementation based on LeNet-5 paper (1998) which is in public domain.

## 🙏 Acknowledgments

- **Yann LeCun** - LeNet-5 architecture và MNIST dataset
- **Andrew Ng** - Deep Learning course với clear explanations
- **Original implementer** - LeNet-from-Scratch notebook structure

---

**Last updated**: November 2025  
**Python version**: 3.6+ (tested on 3.6, 3.9, 3.11)  
**OS**: Windows 10 (should work on Linux/Mac with minor path changes)
