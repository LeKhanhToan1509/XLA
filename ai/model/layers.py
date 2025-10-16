"""
Neural Network Layers Implementation
===================================

Các layers cơ bản cho Deep Learning:
1. Convolutional Layer - Trích xuất features từ images
2. Batch Normalization - Ổn định quá trình training
3. ReLU Activation - Non-linear activation function
4. Max Pooling - Downsampling và translation invariance
5. Fully Connected - Classification layer
6. Softmax - Probability distribution
7. Cross Entropy Loss - Loss function cho classification
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️  Using CPU (NumPy)")
    GPU_AVAILABLE = False
    
from utils.utils import im2col, col2im, get_indices

class Conv:
    """
    Convolutional Layer Implementation
    =================================
    
    Công thức Convolution:
    Y[i,j] = Σ(k=0 to f-1) Σ(l=0 to f-1) Σ(c=0 to C-1) W[f,c,k,l] * X[c,i*s+k,j*s+l] + b[f]
    
    Trong đó:
    - W: Filter weights (nb_filters, nb_channels, filter_size, filter_size)
    - X: Input feature maps (batch_size, nb_channels, height, width)
    - b: Bias terms (nb_filters,)
    - s: Stride
    - f: Filter size
    
    Đầu vào:
    - X: Input tensor (N, C, H, W)
    
    Đầu ra:
    - Y: Output tensor (N, F, H', W') 
    Với H' = (H + 2*pad - filter_size)/stride + 1
         W' = (W + 2*pad - filter_size)/stride + 1
    """
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding
        self.W = {'val': np.random.randn(self.n_F, self.n_C, self.f, self.f) * np.sqrt(2. / (self.n_C * self.f * self.f)),
                  'grad': np.zeros((self.n_F, self.n_C, self.f, self.f))}
        self.b = {'val': np.zeros((self.n_F,)), 'grad': np.zeros((self.n_F,))}
        self.cache = None

    def forward(self, X):
        """
        Forward pass của Convolutional Layer
        
        Sử dụng im2col để tối ưu hóa convolution:
        1. Chuyển đổi X thành ma trận 2D sử dụng im2col
        2. Reshape weights thành ma trận 2D
        3. Thực hiện matrix multiplication: W @ X_col + b
        4. Reshape kết quả về tensor 4D
        
        Đầu vào:
        - X: Input tensor (batch_size, channels, height, width)
        
        Đầu ra:
        - out: Convolved features (batch_size, nb_filters, out_height, out_width)
        """
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        
        # Tính output dimensions
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1
        
        # Im2col transformation để tối ưu convolution
        X_col = im2col(X, self.f, self.f, self.s, self.p)  # (filter_size^2 * channels, out_H * out_W * batch_size)
        w_col = self.W['val'].reshape((self.n_F, -1))  # (nb_filters, filter_size^2 * channels)
        b_col = self.b['val'].reshape(-1, 1)  # (nb_filters, 1)
        
        # Matrix multiplication: convolution thành dot product
        out = w_col @ X_col + b_col  # (nb_filters, out_H * out_W * batch_size)
        
        # Reshape về tensor 4D
        out = np.array(np.hsplit(out, m)).reshape((m, self.n_F, n_H, n_W))
        
        # Cache cho backward pass
        self.cache = (X, X_col, w_col)
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        m = X.shape[0]
        self.b['grad'] = np.sum(dout, axis=(0, 2, 3))
        dout_reshaped = dout.reshape(m, self.n_F, -1)
        dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
        dout = np.array(np.vsplit(dout, m))
        dout = np.concatenate(dout, axis=-1)
        dX_col = w_col.T @ dout
        dw_col = dout @ X_col.T
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        self.W['grad'] = dw_col.reshape((self.n_F, self.n_C, self.f, self.f))
        return dX

class BatchNorm:
    """
    Batch Normalization Layer Implementation
    ======================================
    
    Công thức Batch Normalization:
    1. μ = (1/m) * Σ(xi)  - Mean của batch
    2. σ² = (1/m) * Σ(xi - μ)²  - Variance của batch  
    3. x̂i = (xi - μ) / sqrt(σ² + ε)  - Normalization
    4. yi = γ * x̂i + β  - Scale and shift
    
    Trong đó:
    - μ, σ²: Batch statistics
    - γ, β: Learnable parameters (scale, shift)
    - ε: Small constant để tránh division by zero
    
    Đầu vào:
    - X: Feature maps (batch_size, num_features, height, width)
    
    Đầu ra:
    - Y: Normalized features cùng shape với X
    
    Training vs Inference:
    - Training: Sử dụng batch statistics
    - Inference: Sử dụng moving average statistics
    """
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = {'val': np.ones((1, num_features, 1, 1)), 'grad': np.zeros((1, num_features, 1, 1))}
        self.beta = {'val': np.zeros((1, num_features, 1, 1)), 'grad': np.zeros((1, num_features, 1, 1))}
        self.moving_mean = np.zeros((num_features,))
        self.moving_var = np.ones((num_features,))
        self.cache = None
        self.use_moving_avg = False  # Training mode

    def forward(self, X):
        """
        Forward pass của Batch Normalization
        
        Training mode:
        1. Tính mean và variance từ current batch
        2. Update moving averages cho inference
        3. Normalize: (X - mean) / sqrt(var + eps)
        4. Scale and shift: gamma * X_norm + beta
        
        Inference mode:
        1. Sử dụng stored moving averages
        2. Normalize và scale/shift tương tự
        
        Đầu vào:
        - X: Input tensor (batch_size, num_features, height, width)
        
        Đầu ra:
        - out: Normalized tensor cùng shape với X
        """
        if self.use_moving_avg:  # Inference mode
            # Sử dụng moving averages đã học được
            mean = self.moving_mean.reshape(1, -1, 1, 1)
            var = self.moving_var.reshape(1, -1, 1, 1)
        else:  # Training mode
            m, n_C, n_H, n_W = X.shape
            
            # Tính batch statistics (mean, var theo channels)
            mean = np.mean(X, axis=(0, 2, 3)).reshape(1, -1, 1, 1)  # Shape: (1, C, 1, 1)
            var = np.var(X, axis=(0, 2, 3)).reshape(1, -1, 1, 1)   # Shape: (1, C, 1, 1)
            
            # Update moving averages với momentum = 0.9
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean.flatten()
            self.moving_var = 0.9 * self.moving_var + 0.1 * var.flatten()
            
            # Cache cho backward pass
            self.cache = (X, mean, var)
        
        # Batch normalization: (X - μ) / sqrt(σ² + ε)
        X_centered = X - mean
        X_norm = X_centered / np.sqrt(var + self.eps)
        
        # Scale and shift: γ * X̂ + β
        out = self.gamma['val'] * X_norm + self.beta['val']
        return out

    def backward(self, dout):
        if self.cache is None:
            return dout
        X, mean, var = self.cache
        m, n_C, n_H, n_W = dout.shape
        batch_size = m * n_H * n_W
        X_norm = (X - mean) / np.sqrt(var + self.eps)
        dX_norm = dout * self.gamma['val']
        dvar = np.sum(dX_norm * X_norm, axis=(0,2,3)) * -0.5 * (var + self.eps)**(-1.5)
        dmean = np.sum(dX_norm * -1 / np.sqrt(var + self.eps), axis=(0,2,3))
        dgamma = np.sum(dout * X_norm, axis=(0,2,3)).reshape(1,-1,1,1)
        dbeta = np.sum(dout, axis=(0,2,3)).reshape(1,-1,1,1)
        self.gamma['grad'] = dgamma
        self.beta['grad'] = dbeta
        dX = (dX_norm / np.sqrt(var + self.eps).reshape(1,-1,1,1)) + (dvar * 2 * (X - mean) / batch_size) + (dmean / batch_size)
        return dX

class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation Function
    ==============================================
    
    Công thức ReLU:
    f(x) = max(0, x) = {x if x > 0, 0 if x ≤ 0}
    
    Đạo hàm:
    f'(x) = {1 if x > 0, 0 if x ≤ 0}
    
    Ưu điểm:
    - Tránh vanishing gradient problem
    - Tính toán nhanh
    - Sparsity (nhiều neurons = 0)
    
    Đầu vào:
    - X: Input tensor bất kỳ shape
    
    Đầu ra:
    - Output: Tensor cùng shape, các giá trị âm thành 0
    """
    def __init__(self):
        self.cache = None

    def forward(self, X):
        """
        Forward pass: f(x) = max(0, x)
        
        Đầu vào:
        - X: Input tensor
        
        Đầu ra:
        - Output: max(0, X) element-wise
        """
        self.cache = X  # Lưu input cho backward pass
        return np.maximum(0, X)

    def backward(self, dout):
        """
        Backward pass: gradient = dout * (input > 0)
        
        Đầu vào:
        - dout: Upstream gradients
        
        Đầu ra:
        - dx: Gradients w.r.t input
        """
        return dout * (self.cache > 0)  # Gradient = 1 nếu input > 0, else 0

class MaxPool:
    """
    Max Pooling Layer Implementation
    ===============================
    
    Công thức Max Pooling:
    Y[i,j] = max(X[i*s:i*s+f, j*s:j*s+f])
    
    Trong đó:
    - f: Filter size (pooling window)
    - s: Stride
    - X: Input feature maps
    
    Chức năng:
    - Downsampling để giảm spatial dimensions
    - Translation invariance
    - Giảm computational cost
    - Feature selection (chọn features mạnh nhất)
    
    Đầu vào:
    - X: Feature maps (batch_size, channels, height, width)
    
    Đầu ra:
    - Y: Pooled features (batch_size, channels, height', width')
    Với height' = (height + 2*pad - filter_size)/stride + 1
         width' = (width + 2*pad - filter_size)/stride + 1
    """
    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None
        self.argmax_cache = None

    def forward(self, X):
        self.cache = X
        m, n_C, n_H_prev, n_W_prev = X.shape
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1
        
        X_col = im2col(X, self.f, self.f, self.s, self.p).reshape(m * n_C, n_H * n_W, self.f**2)
        out = np.max(X_col, axis=2).reshape(m, n_C, n_H, n_W)
        self.argmax_cache = np.argmax(X_col, axis=2)
        return out

    def backward(self, dout):
        m, n_C, n_H, n_W = dout.shape
        dout_col = dout.reshape(m * n_C, n_H * n_W, 1)
        dX_col = np.zeros((m * n_C, n_H * n_W, self.f**2))
        dX_col[np.arange(dX_col.shape[0])[:, None], np.arange(dX_col.shape[1])[:, None], self.argmax_cache] = dout_col
        dX_col = dX_col.reshape(-1, self.f * self.f)
        dX = col2im(dX_col.T, self.cache.shape, self.f, self.f, self.s, self.p)
        return dX

class Fc:
    """
    Fully Connected (Dense) Layer Implementation
    ===========================================
    
    Công thức Fully Connected:
    Y = X @ W + b
    
    Trong đó:
     
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = {'val': np.random.randn(input_size, output_size) * np.sqrt(2. / input_size), 'grad': np.zeros((input_size, output_size))}
        self.b = {'val': np.zeros((1, output_size)), 'grad': np.zeros((1, output_size))}
        self.cache = None

    def forward(self, X):
        self.cache = X
        m = X.shape[0]
        return np.dot(X.reshape(m, -1), self.W['val']) + self.b['val']

    def backward(self, dout):
        m = self.cache.shape[0]
        self.W['grad'] = np.dot(self.cache.reshape(m, -1).T, dout) / m
        self.b['grad'] = np.sum(dout, axis=0, keepdims=True) / m
        dX = np.dot(dout, self.W['val'].T).reshape(self.cache.shape)
        return dX

class Softmax:
    """
    Softmax Activation Function
    ==========================
    
    Công thức Softmax:
    softmax(xi) = exp(xi) / Σ(j=1 to K) exp(xj)
    
    Với numerical stability:
    softmax(xi) = exp(xi - max(x)) / Σ(j=1 to K) exp(xj - max(x))
    
    Tính chất:
    - Σ softmax(xi) = 1 (tạo probability distribution)
    - 0 ≤ softmax(xi) ≤ 1
    - Differentiable
    
    Đầu vào:
    - X: Logits (batch_size, num_classes)
    
    Đầu ra:
    - Probabilities: (batch_size, num_classes)
    
    Gradient (kết hợp với CrossEntropy):
    ∂L/∂xi = yi_pred - yi_true
    """
    def __init__(self):
        pass

    def forward(self, X):
        """
        Forward pass với numerical stability
        
        Đầu vào:
        - X: Logits (batch_size, num_classes)
        
        Đầu ra:
        - Probabilities: Same shape với X
        """
        # Subtract max để tránh overflow (numerical stability)
        e_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def backward(self, y_pred, y):
        """
        Backward pass (kết hợp với CrossEntropy loss)
        
        Gradient của Softmax + CrossEntropy:
        ∂L/∂logits = y_pred - y_true
        
        Đầu vào:
        - y_pred: Predicted probabilities
        - y: True labels (one-hot)
        
        Đầu ra:
        - Gradients w.r.t logits
        """
        return y_pred - y

class CrossEntropyLoss:
    """
    Cross Entropy Loss Function
    ==========================
    
    Công thức Cross Entropy Loss:
    L = -1/m * Σ(i=1 to m) Σ(j=1 to K) yij * log(ŷij)
    
    Trong đó:
    - m: Batch size
    - K: Số classes
    - yij: True label (one-hot encoded)
    - ŷij: Predicted probability
    
    Tính chất:
    - Phù hợp cho multi-class classification
    - Gradient tốt cho training
    - Penalty cao cho confident wrong predictions
    
    Đầu vào:
    - y_pred: Predicted probabilities (batch_size, num_classes)
    - y: True labels one-hot (batch_size, num_classes)
    
    Đầu ra:
    - Loss: Scalar value
    """
    def __init__(self):
        pass
    
    def get(self, y_pred, y):
        """
        Tính Cross Entropy Loss
        
        Đầu vào:
        - y_pred: Predicted probabilities
        - y: True labels (one-hot)
        
        Đầu ra:
        - loss: Average cross entropy loss
        """
        m = y.shape[0]  # Batch size
        # Thêm epsilon để tránh log(0)
        return -np.sum(y * np.log(y_pred + 1e-8)) / m