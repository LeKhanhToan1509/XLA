"""
Optimization Algorithms Module
=============================

Adam Optimizer Implementation
============================

Công thức Adam Optimizer:
1. mt = β1 * m(t-1) + (1-β1) * gt     (First moment estimate)
2. vt = β2 * v(t-1) + (1-β2) * gt²    (Second moment estimate)  
3. m̂t = mt / (1 - β1^t)               (Bias-corrected first moment)
4. v̂t = vt / (1 - β2^t)               (Bias-corrected second moment)
5. θt = θ(t-1) - α * m̂t / (√v̂t + ε)   (Parameter update)

Trong đó:
- gt: Gradients tại step t
- mt, vt: Moving averages của gradients và squared gradients
- β1, β2: Decay rates (thường 0.9, 0.999)
- α: Learning rate
- ε: Small constant để tránh division by zero

Ưu điểm Adam:
- Adaptive learning rates cho từng parameter
- Hiệu quả với sparse gradients
- Ít sensitive với hyperparameters
- Converge nhanh
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️  Using CPU (NumPy)")
    GPU_AVAILABLE = False

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Khởi tạo Adam Optimizer
        
        Đầu vào:
        - lr: Learning rate (α) - tốc độ học
        - beta1: Decay rate cho first moment (thường 0.9)
        - beta2: Decay rate cho second moment (thường 0.999)  
        - epsilon: Small constant để numerical stability (thường 1e-8)
        
        State variables:
        - t: Time step counter
        - m: First moment estimates (moving average của gradients)
        - v: Second moment estimates (moving average của squared gradients)
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.m = None  # First moment estimates
        self.v = None  # Second moment estimates

    def update(self, params, grads):
        """
        Cập nhật parameters sử dụng Adam algorithm
        
        Thực hiện các bước:
        1. Increment time step
        2. Update first và second moment estimates
        3. Compute bias-corrected estimates  
        4. Update parameters
        
        Đầu vào:
        - params: Dictionary chứa parameters cần update
                  Format: {'W_layer1': weight_matrix, 'b_layer1': bias_vector, ...}
        - grads: Dictionary chứa gradients tương ứng
                 Format: {'dW_layer1': grad_matrix, 'db_layer1': grad_vector, ...}
        
        Đầu ra:
        - params: Updated parameters dictionary
        """
        self.t += 1  # Increment time step
        
        # Khởi tạo moment estimates lần đầu
        if self.m is None:
            self.m = {k: np.zeros_like(v) for k, v in params.items()}
            self.v = {k: np.zeros_like(v) for k, v in params.items()}
        
        # Cập nhật từng parameter
        for key in params:
            # Lấy gradient tương ứng (key matching phải chính xác)
            # grads và params dùng cùng key format: 'W_12345', 'b_12345', etc.
            if key not in grads:
                # Debug: print warning if gradient not found
                if self.t == 1:  # Only print once at first update
                    print(f"⚠️  Warning: No gradient found for parameter '{key}' - skipping update")
                continue
            
            grad = grads[key]
            
            # Skip if gradient is None or all zeros
            if grad is None or np.sum(np.abs(grad)) == 0:
                if self.t == 1:
                    print(f"⚠️  Warning: Zero gradient for parameter '{key}' - skipping update")
                continue
            
            # Update biased first moment estimate: mt = β1*m(t-1) + (1-β1)*gt
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate: vt = β2*v(t-1) + (1-β2)*gt²
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate: m̂t = mt/(1-β1^t)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate: v̂t = vt/(1-β2^t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters: θt = θ(t-1) - α*m̂t/(√v̂t + ε)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        return params