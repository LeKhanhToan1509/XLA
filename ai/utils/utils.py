"""
Utility Functions for Convolutional Operations
=============================================

Im2Col và Col2Im Implementation
==============================

Im2col (Image to Column):
- Chuyển đổi convolution operation thành matrix multiplication
- Từ sliding window convolution → efficient GEMM operation
- Tối ưu performance nhờ vectorized operations

Col2im (Column to Image):  
- Inverse operation của im2col
- Dùng trong backward pass để reconstruct gradients
- Accumulate overlapping regions

Tại sao dùng Im2col:
1. Performance: Matrix multiplication được optimize rất tốt trên modern hardware
2. Memory locality: Better cache utilization
3. Parallelization: Dễ dàng parallelize matrix operations
4. Code simplicity: Convolution trở thành simple dot product

Trade-offs:
- Memory overhead: Duplicate data trong overlapping windows
- Temporary storage: Cần extra memory cho intermediate results
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️  Using CPU (NumPy)")
    GPU_AVAILABLE = False

def get_indices(X_shape, HF, WF, stride, pad):
    """
    Tính toán indices cho Im2col operation
    
    Mục đích: Tạo index arrays để extract patches từ input tensor
    
    Đầu vào:
    - X_shape: Shape của input (batch_size, channels, height, width)
    - HF, WF: Filter dimensions (height, width)
    - stride: Stride của convolution
    - pad: Padding size
    
    Đầu ra:
    - i, j, d: Index arrays cho height, width, và depth dimensions
    
    Logic:
    1. Tính output dimensions sau convolution
    2. Tạo indices cho mỗi filter position
    3. Replicate cho tất cả channels và output positions
    """
    m, n_C, n_H, n_W = X_shape
    
    # Tính output dimensions
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
  
    # Height indices: Tạo pattern cho filter heights
    level1 = np.repeat(np.arange(HF), WF)  # [0,0,0, 1,1,1, 2,2,2] cho HF=3,WF=3
    level1 = np.tile(level1, n_C)  # Replicate cho tất cả channels
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)  # Stride offsets cho output positions
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)  # Broadcasting để tạo full index matrix

    # Width indices: Tương tự cho width dimension
    slide1 = np.tile(np.arange(WF), HF)  # [0,1,2, 0,1,2, 0,1,2] cho HF=3,WF=3
    slide1 = np.tile(slide1, n_C)  # Replicate cho tất cả channels
    everySlides = stride * np.tile(np.arange(out_w), out_h)  # Stride offsets cho output positions
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)  # Broadcasting để tạo full index matrix

    # Depth (channel) indices
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)  # [0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, ...]

    return i, j, d

def im2col(X, HF, WF, stride, pad):
    """
    Im2col transformation cho efficient convolution
    
    Chuyển đổi convolution operation thành matrix multiplication:
    Convolution: Y = W * X (sliding window)
    Im2col: Y = W_reshaped @ X_col (matrix multiplication)
    
    Đầu vào:
    - X: Input tensor (batch_size, channels, height, width)
    - HF, WF: Filter dimensions
    - stride: Convolution stride
    - pad: Padding size
    
    Đầu ra:
    - cols: Columnized matrix (filter_area * channels, output_area * batch_size)
    
    Process:
    1. Pad input tensor
    2. Extract all patches theo sliding window pattern
    3. Reshape thành matrix format cho GEMM
    """
    # Apply padding
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    
    # Get indices cho patch extraction
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    
    # Extract patches sử dụng advanced indexing
    cols = X_padded[:, d, i, j]  # Shape: (batch_size, filter_area*channels, output_area)
    
    # Concatenate along batch dimension
    cols = np.concatenate(cols, axis=-1)  # Shape: (filter_area*channels, output_area*batch_size)
    
    return cols

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    """
    Col2im transformation - inverse của im2col
    
    Sử dụng trong backward pass để reconstruct gradient tensor
    từ columnized gradients về original tensor shape
    
    Đầu vào:
    - dX_col: Gradients trong col format (filter_area*channels, output_area*batch_size)
    - X_shape: Shape của original input tensor
    - HF, WF: Filter dimensions  
    - stride: Convolution stride
    - pad: Padding size
    
    Đầu ra:
    - dX: Reconstructed gradients (batch_size, channels, height, width)
    
    Process:
    1. Tạo zero tensor với padded dimensions
    2. Accumulate gradients từ overlapping patches
    3. Remove padding để về original size
    
    Lưu ý: Overlapping regions được cộng dồn (accumulate)
    """
    N, D, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    
    # Tạo padded output tensor
    X_padded = np.zeros((N, D, H_padded, W_padded))
    
    # Get same indices như im2col
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    
    # Reshape gradients back to batch format
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))  # Split theo batch dimension
    
    # Accumulate gradients vào positions tương ứng
    # np.add.at handles overlapping indices by adding values
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    
    # Remove padding để về original size
    if pad == 0:
        return X_padded
    return X_padded[:, :, pad:-pad, pad:-pad]