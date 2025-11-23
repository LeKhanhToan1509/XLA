# Streamlit App - LeNet-5 Digit & Shape Classifier

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements_app.txt
```

### 2. Run the app

```bash
cd e:\PTIT\XLA
streamlit run ai\app.py
```

App sẽ mở tại: `http://localhost:8501`

## 🎨 Features

### Mode 1: Realtime Drawing
- Vẽ trực tiếp trên canvas một chữ số hoặc hình
- Adjust brush size
- Nhận dạng số (0-9) hoặc hình (circle, square, triangle)
- Clear và vẽ lại dễ dàng

### Mode 2: Detect Numbers
- **Vẽ hoặc upload ảnh chứa nhiều số (0-9)**
- Tự động phát hiện và tách từng số trong ảnh
- **Advanced preprocessing**:
  - Adaptive thresholding (tự động điều chỉnh)
  - Auto detect và invert màu (chữ trắng/đen)
  - Morphological operations để loại bỏ noise
  - CLAHE enhancement cho contrast tốt hơn
  - Smart padding và centering
- Nhận dạng tất cả các số được tìm thấy
- Hiển thị confidence score cho từng số
- Xem preprocessing steps (tùy chọn)

### Mode 3: Upload File
- Upload ảnh PNG, JPG, JPEG, BMP
- Auto resize về 28×28
- Show original và preprocessed image
- Auto nhận dạng

## 📊 Display

- **Detected Numbers**: Hiển thị tất cả các số được phát hiện trong hình
- **Confidence Scores**: Bar chart cho từng số được phát hiện
- **Individual Detections**: Hiển thị từng số được cắt ra và nhận dạng
- **Preprocessed Image**: 28×28 như model input
- **Summary**: Tổng số và danh sách các số được nhận dạng

## 💡 Use Cases

### 1. Single Digit Recognition (Mode 1)
Vẽ một số hoặc hình đơn lẻ để nhận dạng nhanh

### 2. Multiple Numbers Detection (Mode 2)
- Viết nhiều số trên cùng một ảnh
- Ví dụ: viết "1234" hoặc "2025"
- App sẽ tự động tách và nhận dạng từng số
- Hữu ích cho:
  - Nhận dạng mã PIN
  - Đọc số điện thoại viết tay
  - Phát hiện nhiều số trong một hình ảnh

### 3. Batch Image Processing (Mode 3)
Upload ảnh có sẵn để test model

## 🔧 Customization

### Change model path

Trong sidebar, edit "Model Path":
```
ai/models/model_weights_final.pkl
```

Hoặc chọn epoch khác:
```
ai/models/model_weights_19.pkl
```

### Adjust canvas

Trong code `ai/app.py`:
```python
canvas_size = 280  # Canvas size (pixels)
stroke_width = 20  # Default brush size
```

## 🐛 Troubleshooting

### streamlit-drawable-canvas not found

```bash
pip install streamlit-drawable-canvas
```

### Model not loading

Check:
1. Model path đúng: `ai/models/model_weights_final.pkl`
2. Model file tồn tại
3. Model có đúng format (dict với keys: C1_weight, C3_wb, etc.)

### Import errors

Run từ root directory:
```bash
cd e:\PTIT\XLA
streamlit run ai\app.py
```

Không chạy từ `ai/`:
```bash
# SAI:
cd ai
streamlit run app.py  # Sẽ lỗi import
```

### Canvas không vẽ được

- Thử refresh browser
- Check browser console (F12) for errors
- Thử browser khác (Chrome recommended)

## 📸 Screenshots

### Realtime Drawing Mode
- Draw canvas với adjustable brush
- Live prediction các số có trong hình với confidence scores

### Upload File Mode
- Drag & drop hoặc browse files
- Auto nhận dạng các số sau khi upload
- Display original + preprocessed images

## 🎯 Tips for Best Results

### Realtime Drawing Mode:
1. Vẽ số rõ ràng ở giữa canvas
2. Kích thước vừa phải (không quá to hoặc nhỏ)
3. Vẽ rõ ràng, liên tục (không đứt nét)
4. Brush size 15-25 cho độ rõ nét tốt nhất
5. Sử dụng "Smart Resize" cho kết quả tốt hơn

### Detect Numbers Mode:
1. **Viết các số cách nhau** để dễ phát hiện
2. Kích thước số nên đồng đều
3. Contrast cao (chữ đậm, nền sáng hoặc ngược lại)
4. Không viết chồng lên nhau
5. Bật "Show Preprocessing Steps" để debug
6. **Khoảng cách tối thiểu**: 10-20 pixels giữa các số
7. **Kích thước tối thiểu**: mỗi số ít nhất 15×15 pixels

### Upload Mode:
1. Dùng ảnh có contrast cao (trắng trên đen hoặc ngược lại)
2. Nội dung số ở giữa ảnh
3. Size gốc 28×28 hoặc bất kỳ (sẽ auto resize)
4. Format: PNG với transparent background hoặc white/black background
5. Ảnh chứa 1 số hoặc nhiều số (sẽ được phát hiện và nhận dạng)

## 📝 Example Images

Test với ảnh từ `data/test/`:
```
data/test/0/mnist_*.png
data/test/1/mnist_*.png
data/test/2/mnist_*.png
...
data/test/9/mnist_*.png
```

## 🔄 Advanced Usage

### Multiple Numbers Detection Algorithm

App sử dụng pipeline xử lý ảnh tối ưu:

#### 1. **Preprocessing Pipeline:**
```
Input Image
    ↓
Grayscale Conversion
    ↓
Auto Invert Detection (nếu nền tối)
    ↓
Gaussian Blur (giảm noise)
    ↓
Adaptive Thresholding (tự động điều chỉnh threshold)
    ↓
Morphological Operations (clean up)
    ↓
Contour Detection
```

#### 2. **Character Segmentation:**
- **Contour filtering**: Loại bỏ noise dựa trên size và aspect ratio
- **Bounding box extraction**: Tìm vùng chứa mỗi số
- **Sort left-to-right**: Sắp xếp theo thứ tự từ trái sang phải

#### 3. **Individual Character Processing:**
```
Extracted Region
    ↓
Dynamic Padding (10-15% of size)
    ↓
Square Centering (white background)
    ↓
CLAHE Enhancement (contrast)
    ↓
Gaussian Smoothing
    ↓
Resize to 28×28 (INTER_AREA)
    ↓
Otsu Thresholding (MNIST-style)
```

#### 4. **Why This Works Best:**
- **Adaptive Threshold**: Tự động điều chỉnh cho từng vùng của ảnh
- **CLAHE**: Cải thiện contrast cục bộ
- **INTER_AREA**: Interpolation tốt nhất cho downsampling
- **Morphological Ops**: Loại bỏ noise nhỏ và lấp lỗ nhỏ
- **Smart Padding**: Đảm bảo số không bị cắt xén

### Multiple numbers detection

App có thể nhận dạng nhiều số trong cùng một hình ảnh.

### Batch prediction (TODO)

Upload multiple images và predict tất cả.

### Export results (TODO)

Save predictions với confidence scores và vị trí các số.

### Webcam input (TODO)

Dùng webcam để capture và predict realtime.

## 📚 Technology Stack

- **Streamlit**: Web app framework
- **streamlit-drawable-canvas**: Canvas component for drawing
- **OpenCV**: Image processing và số detection
- **NumPy**: Array operations
- **Pillow**: Image I/O

## 🤝 Contributing

Improvements welcome:
- Add multi-number detection trong cùng ảnh
- Add bounding box visualization
- Add batch prediction
- Add webcam support
- Improve canvas UX
- Add more visualization (feature maps, gradients)
- Mobile responsive design
