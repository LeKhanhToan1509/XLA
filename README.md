# 🧠 LeNet-5 from Scratch - MNIST & Shape Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **LeNet-5 CNN implementation từ đầu** với NumPy thuần (không dùng TensorFlow/PyTorch) để phân loại chữ số và hình học

## 📸 Demo

### Streamlit Web App
![Demo App](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=LeNet-5+Demo+App)

**3 chế độ hoạt động:**
- 🎨 **Realtime Drawing** - Vẽ trực tiếp và nhận diện
- 🔍 **Detect Numbers** - Phát hiện nhiều số trong một ảnh
- 📁 **Upload File** - Upload ảnh và nhận diện tự động

## ✨ Tính năng

- ✅ **100% From-scratch**: Chỉ dùng NumPy, không framework
- ✅ **13 Classes**: 0-9 (MNIST) + Circle, Square, Triangle
- ✅ **SDLM Optimizer**: Stochastic Diagonal Levenberg-Marquardt
- ✅ **RBF Output Layer**: 7×12 bitmap patterns
- ✅ **Advanced Preprocessing**: Adaptive threshold, CLAHE, morphological operations
- ✅ **Web Interface**: Streamlit app với 3 chế độ
- ✅ **Model Checkpointing**: Auto-save mỗi epoch

## 🚀 Quick Start

### 1. Cài đặt dependencies

```bash
# Clone repository
git clone https://github.com/LeKhanhToan1509/XLA.git
cd XLA

# Install dependencies cho training
pip install -r ai/requirements.txt

# Install dependencies cho web app
pip install -r ai/requirements_app.txt
```

### 2. Chạy Web App

```bash
streamlit run ai/app.py
```

App sẽ mở tại: `http://localhost:8501`

### 3. Train Model (Optional)

```bash
cd ai
python main.py
```

## 📁 Cấu trúc Project

```
XLA/
├── ai/                          # Main application
│   ├── app.py                   # Streamlit web app
│   ├── main.py                  # Training entry point
│   ├── README.md                # Chi tiết kỹ thuật
│   ├── APP_README.md            # Hướng dẫn sử dụng app
│   ├── requirements.txt         # Dependencies cho training
│   ├── requirements_app.txt     # Dependencies cho web app
│   ├── models/                  # Model checkpoints (*.pkl)
│   └── utils/                   # Core implementation
│       ├── LayerObjects.py      # LeNet-5 architecture
│       ├── Convolution_util.py  # Conv operations
│       ├── Pooling_util.py      # Pooling operations
│       ├── Activation_util.py   # Activation functions
│       ├── RBF_initial_weight.py # RBF layer weights
│       ├── utils_func.py        # Utilities
│       └── generate_shape.py    # Shape generation
│
├── data/                        # Dataset (not in repo)
│   ├── train/                   # Training data
│   ├── test/                    # Test data
│   └── val/                     # Validation data
│
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 📊 Architecture

### LeNet-5 Structure

```
Input (32×32) 
    ↓
C1: Conv 6@5×5 → 6@28×28
    ↓
S2: AvgPool 2×2 → 6@14×14
    ↓
C3: Conv 16@5×5 → 16@10×10 (custom mapping)
    ↓
S4: AvgPool 2×2 → 16@5×5
    ↓
C5: Conv 120@5×5 → 120@1×1
    ↓
F6: Fully Connected → 84
    ↓
RBF: Output Layer → 13 classes
```

### Key Features

- **C3 Custom Mapping**: Theo paper gốc LeCun (không phải full connection)
- **RBF Layer**: 7×12 ASCII-style bitmap patterns
- **SDLM Optimizer**: Adaptive learning rate cho từng layer
- **Squash Activation**: `1.7159 * tanh(2x/3)`

## 🎯 Performance

| Metric | Value |
|--------|-------|
| Dataset | MNIST + Shapes |
| Classes | 13 |
| Accuracy | ~95%+ |
| Training Time | ~2-3 hours (20 epochs) |
| Model Size | ~500KB |

## 🎨 Web App Features

### Mode 1: Realtime Drawing
- Vẽ trực tiếp trên canvas
- Smart resize với auto-centering
- Real-time prediction với confidence scores

### Mode 2: Detect Numbers
- Upload hoặc vẽ ảnh chứa nhiều số
- Tự động phát hiện và tách từng số
- Advanced preprocessing pipeline:
  - Adaptive thresholding
  - Auto invert detection
  - Morphological operations
  - CLAHE enhancement

### Mode 3: Upload File
- Drag & drop hoặc browse files
- Support PNG, JPG, JPEG, BMP
- Auto resize và preprocess

## 📖 Documentation

- **[ai/README.md](ai/README.md)** - Chi tiết kỹ thuật, architecture, training
- **[ai/APP_README.md](ai/APP_README.md)** - Hướng dẫn sử dụng web app
- **[ai/utils/](ai/utils/)** - Source code implementation

## 🛠️ Technologies

- **NumPy** - Core implementation
- **OpenCV** - Image preprocessing
- **Streamlit** - Web interface
- **Pillow** - Image handling
- **Pickle** - Model serialization

## 📝 Dataset

### MNIST Digits (0-9)
- Training: ~60,000 images
- Test: ~10,000 images
- Format: 28×28 grayscale

### Generated Shapes (Circle, Square, Triangle)
- Training: 1,000 per class
- Test: 200 per class
- Format: 28×28 grayscale
- Auto-generated với variations

**Download dataset:**
```bash
# MNIST sẽ tự động download khi training
# Shapes sẽ được generate tự động
python ai/utils/generate_shape.py
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Lê Khánh Toàn** - [LeKhanhToan1509](https://github.com/LeKhanhToan1509)

## 🙏 Acknowledgments

- Yann LeCun et al. - Original LeNet-5 paper
- MNIST dataset creators
- NumPy community

## 📞 Contact

- GitHub: [@LeKhanhToan1509](https://github.com/LeKhanhToan1509)
- Repository: [XLA](https://github.com/LeKhanhToan1509/XLA)

---

⭐ **Star this repo if you find it helpful!**
