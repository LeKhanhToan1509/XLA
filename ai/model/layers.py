# ai/model/layers.py
"""
Layers cơ bản: Conv, BatchNorm, ReLU, MaxPool, Fc, Softmax, CrossEntropyLoss.
Sử dụng np_local từ utils và manual_pad.
"""

import numpy as np

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

# Import np_local từ utils
from ..utils.utils import np_local, manual_pad, _is_cupy

from ..utils.utils import im2col, col2im, get_indices  # Giữ nguyên

class Conv:
    def __init__(self, nb_filters, filter_size, nb_channels, stride=1, padding=0):
        self.n_F = nb_filters
        self.f = filter_size
        self.n_C = nb_channels
        self.s = stride
        self.p = padding
        self.W = {'val': xp.random.randn(self.n_F, self.n_C, self.f, self.f) * xp.sqrt(2. / (self.n_C * self.f * self.f)),
                  'grad': xp.zeros((self.n_F, self.n_C, self.f, self.f))}
        self.b = {'val': xp.zeros((self.n_F,)), 'grad': xp.zeros((self.n_F,))}
        self.cache = None

    def forward(self, X):
        m, n_C_prev, n_H_prev, n_W_prev = X.shape
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1
        X_col = im2col(X, self.f, self.f, self.s, self.p)
        w_col = self.W['val'].reshape((self.n_F, -1))
        b_col = self.b['val'].reshape(-1, 1)
        out_temp = w_col @ X_col + b_col
        # Tối ưu: reshape trực tiếp thay vì hsplit
        out = out_temp.reshape(self.n_F, m, n_H, n_W).transpose(1, 0, 2, 3)
        self.cache = (X, X_col, w_col)
        return out

    def backward(self, dout):
        X, X_col, w_col = self.cache
        m = X.shape[0]
        self.b['grad'] = xp.sum(dout, axis=(0, 2, 3))  # Đã zero trước đó, không cần +=
        
        # Reshape dout: (m, n_F, H, W) -> (n_F, m*H*W)
        dout_flat = dout.transpose(1, 0, 2, 3).reshape(self.n_F, -1)
        
        # Compute gradients
        dX_col = w_col.T @ dout_flat
        dw_col = dout_flat @ X_col.T / m
        
        # Convert back to image format
        dX = col2im(dX_col, X.shape, self.f, self.f, self.s, self.p)
        self.W['grad'] = dw_col.reshape((self.n_F, self.n_C, self.f, self.f))
        return dX

class BatchNorm:
    def __init__(self, num_features, eps=1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = {'val': xp.ones((1, num_features, 1, 1)), 'grad': xp.zeros((1, num_features, 1, 1))}
        self.beta = {'val': xp.zeros((1, num_features, 1, 1)), 'grad': xp.zeros((1, num_features, 1, 1))}
        self.moving_mean = xp.zeros((num_features,))
        self.moving_var = xp.ones((num_features,))
        self.cache = None
        self.use_moving_avg = False

    def forward(self, X):
        if self.use_moving_avg:
            mean = self.moving_mean.reshape(1, -1, 1, 1)
            var = self.moving_var.reshape(1, -1, 1, 1)
        else:
            m, n_C, n_H, n_W = X.shape
            mean = xp.mean(X, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            var = xp.var(X, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean.flatten()
            self.moving_var = 0.9 * self.moving_var + 0.1 * var.flatten()
            self.cache = (X, mean, var)
        X_centered = X - mean
        X_norm = X_centered / xp.sqrt(var + self.eps)
        out = self.gamma['val'] * X_norm + self.beta['val']
        return out

    def backward(self, dout):
        if self.cache is None:
            return dout
        X, mean, var = self.cache
        m, n_C, n_H, n_W = dout.shape
        batch_size = m * n_H * n_W
        X_norm = (X - mean) / xp.sqrt(var + self.eps)
        dX_norm = dout * self.gamma['val']
        dvar = xp.sum(dX_norm * (X - mean), axis=(0, 2, 3), keepdims=True) * -0.5 * (var + self.eps)**(-1.5)
        dmean = xp.sum(dX_norm * -1 / xp.sqrt(var + self.eps), axis=(0, 2, 3), keepdims=True)
        dmean += dvar * xp.sum(-2 * (X - mean), axis=(0, 2, 3), keepdims=True) / batch_size
        dX = dX_norm / xp.sqrt(var + self.eps) + dvar * 2 * (X - mean) / batch_size + dmean / batch_size
        self.gamma['grad'] = xp.sum(dout * X_norm, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        self.beta['grad'] = xp.sum(dout, axis=(0, 2, 3)).reshape(1, -1, 1, 1)
        return dX

class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, X):
        self.cache = X.copy()
        return xp.maximum(0, X)

    def backward(self, dout):
        return dout * (self.cache > 0)

class MaxPool:
    def __init__(self, filter_size, stride=1, padding=0):
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.cache = None
        self.argmax_cache = None

    def forward(self, X):
        self.cache = X.copy()
        m, n_C, n_H_prev, n_W_prev = X.shape
        n_H = int((n_H_prev + 2 * self.p - self.f) / self.s) + 1
        n_W = int((n_W_prev + 2 * self.p - self.f) / self.s) + 1
        pad_width = ((0,0), (0,0), (self.p, self.p), (self.p, self.p))
        X_padded = manual_pad(X, pad_width) if self.p > 0 else X
        out = xp.zeros((m, n_C, n_H, n_W))
        self.argmax_cache = xp.zeros((m, n_C, n_H, n_W), dtype=xp.int32)
        for i in range(n_H):
            h_start = i * self.s
            h_end = h_start + self.f  # Fix: Định nghĩa rõ ràng
            for j in range(n_W):
                w_start = j * self.s
                w_end = w_start + self.f  # Fix: Định nghĩa rõ ràng
                window = X_padded[:, :, h_start:h_end, w_start:w_end]
                window_flat = window.reshape(m, n_C, -1)
                out[:, :, i, j] = xp.max(window_flat, axis=2)
                self.argmax_cache[:, :, i, j] = xp.argmax(window_flat, axis=2)
        return out

    def backward(self, dout):
        if self.cache is None:
            return dout
        X = self.cache
        m, n_C, n_H, n_W = dout.shape
        _, _, n_H_prev, n_W_prev = X.shape
        
        # Create padded dX
        if self.p > 0:
            dX_padded = xp.zeros((m, n_C, n_H_prev + 2*self.p, n_W_prev + 2*self.p))
        else:
            dX_padded = xp.zeros((m, n_C, n_H_prev, n_W_prev))
        
        for i in range(n_H):
            h_start = i * self.s
            h_end = h_start + self.f
            for j in range(n_W):
                w_start = j * self.s
                w_end = w_start + self.f
                for b in range(m):
                    for c in range(n_C):
                        max_idx = self.argmax_cache[b, c, i, j]
                        h_offset = max_idx // self.f
                        w_offset = max_idx % self.f
                        dX_padded[b, c, h_start + h_offset, w_start + w_offset] += dout[b, c, i, j]
        
        # Remove padding if needed
        if self.p > 0:
            return dX_padded[:, :, self.p:-self.p, self.p:-self.p]
        else:
            return dX_padded

class Fc:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = {'val': xp.random.randn(input_size, output_size) * xp.sqrt(2. / input_size), 'grad': xp.zeros((input_size, output_size))}
        self.b = {'val': xp.zeros((1, output_size)), 'grad': xp.zeros((1, output_size))}
        self.cache = None

    def forward(self, X):
        self.cache = X.copy()
        m = X.shape[0]
        return xp.dot(X.reshape(m, -1), self.W['val']) + self.b['val']

    def backward(self, dout):
        m = self.cache.shape[0]
        self.W['grad'] = xp.dot(self.cache.reshape(m, -1).T, dout) / m
        self.b['grad'] = xp.sum(dout, axis=0, keepdims=True) / m
        dX = xp.dot(dout, self.W['val'].T).reshape(self.cache.shape)
        return dX

class Softmax:
    def __init__(self):
        pass

    def forward(self, X):
        e_x = xp.exp(X - xp.max(X, axis=1, keepdims=True))
        return e_x / xp.sum(e_x, axis=1, keepdims=True)

    def backward(self, y_pred, y):
        return y_pred - y

class CrossEntropyLoss:
    def __init__(self):
        pass

    def get(self, y_pred, y):
        m = y.shape[0]
        return -xp.sum(y * xp.log(y_pred + 1e-8)) / m
