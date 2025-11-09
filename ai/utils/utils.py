# ai/utils/utils.py
"""
Hàm tiện ích: im2col, col2im, get_indices với manual pad an toàn cho CuPy.
"""

import numpy as np

try:
    import cupy as cp
    _is_cupy = True
    np_local = cp
    print("✅ Sử dụng GPU (CuPy)")
except ImportError:
    _is_cupy = False
    np_local = np
    print("⚠️  Sử dụng CPU (NumPy)")

def manual_pad(X, pad_width):
    """Manual padding an toàn cho CuPy/NumPy, tránh bug constant fill."""
    if _is_cupy:
        # CuPy có bug với pad, dùng workaround
        # Tạo padded array với zeros rồi copy data vào
        m, c, h, w = X.shape
        pad_h_top, pad_h_bot = pad_width[2]
        pad_w_left, pad_w_right = pad_width[3]
        padded = np_local.zeros((m, c, h + pad_h_top + pad_h_bot, w + pad_w_left + pad_w_right), dtype=X.dtype)
        padded[:, :, pad_h_top:pad_h_top+h, pad_w_left:pad_w_left+w] = X
        return padded
    else:
        # NumPy hoạt động bình thường
        return np_local.pad(X, pad_width, mode='constant', constant_values=0)

def get_indices(X_shape, HF, WF, stride, pad):
    m, n_C, n_H, n_W = X_shape
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1
    # Fix: Sử dụng np_local cho consistency
    level1 = np_local.tile(np_local.repeat(np_local.arange(HF), WF), n_C)
    everyLevels = stride * np_local.repeat(np_local.arange(out_h), out_w)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    slide1 = np_local.tile(np_local.tile(np_local.arange(WF), HF), n_C)
    everySlides = stride * np_local.tile(np_local.arange(out_w), out_h)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)
    d = np_local.repeat(np_local.arange(n_C), HF * WF).reshape(-1, 1)
    return i, j, d

def im2col(X, HF, WF, stride, pad):
    if pad > 0:
        pad_width = ((0,0), (0,0), (pad, pad), (pad, pad))
        X_padded = manual_pad(X, pad_width)
    else:
        X_padded = X
    
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    cols = X_padded[:, d, i, j]
    return np_local.concatenate(cols, axis=-1)

def col2im(dX_col, X_shape, HF, WF, stride, pad):
    N, D, H, W = X_shape
    pad_width = ((0,0), (0,0), (pad, pad), (pad, pad))
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np_local.zeros((N, D, H_padded, W_padded))
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    dX_col_reshaped = np_local.array(np_local.hsplit(dX_col, N))
    # add.at an toàn cho cả CuPy/NumPy
    if _is_cupy:
        np_local.add.at(X_padded, (np_local.s_[:], d, i, j), dX_col_reshaped)
    else:
        np.add.at(X_padded, (np.s_[:], d, i, j), dX_col_reshaped)
    return X_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else X_padded