# ai/model/resnet.py
"""
ResNet-18 đầy đủ với fix collect_recursive (skip layers không params như ReLU).
Fallback _last_feature_shape = (1,1) cho input 28x28 sau full layers.
"""

import numpy as np

try:
    import cupy as cp
    xp = cp
except ImportError:
    xp = np

from .layers import Conv, BatchNorm, ReLU, MaxPool, Fc, Softmax, CrossEntropyLoss
from ..configs.config import NUM_CLASSES, INPUT_SHAPE
from ..utils.optimizers import Adam

# Implement collect nếu chưa có trong utils (để độc lập)
def collect_params_recursive(layer, params_dict, prefix=''):
    if isinstance(layer, (Conv, Fc)):
        params_dict[f'{prefix}W_{id(layer)}'] = layer.W['val']
        params_dict[f'{prefix}b_{id(layer)}'] = layer.b['val']
    elif isinstance(layer, BatchNorm):
        params_dict[f'{prefix}gamma_{id(layer)}'] = layer.gamma['val']
        params_dict[f'{prefix}beta_{id(layer)}'] = layer.beta['val']
    # Skip ReLU, MaxPool (không params)
    # Recursive cho BasicBlock
    for attr in ['conv1', 'conv2', 'downsample', 'bn1', 'bn2', 'bn_down']:
        sub_layer = getattr(layer, attr, None)
        if sub_layer is not None and not isinstance(sub_layer, (ReLU, MaxPool)):
            collect_params_recursive(sub_layer, params_dict, f'{prefix}{attr}_')

def collect_grads_recursive(layer, grads_dict, prefix=''):
    if isinstance(layer, (Conv, Fc)):
        grads_dict[f'{prefix}W_{id(layer)}'] = layer.W['grad']
        grads_dict[f'{prefix}b_{id(layer)}'] = layer.b['grad']
    elif isinstance(layer, BatchNorm):
        grads_dict[f'{prefix}gamma_{id(layer)}'] = layer.gamma['grad']
        grads_dict[f'{prefix}beta_{id(layer)}'] = layer.beta['grad']
    # Skip ReLU, MaxPool
    for attr in ['conv1', 'conv2', 'downsample', 'bn1', 'bn2', 'bn_down']:
        sub_layer = getattr(layer, attr, None)
        if sub_layer is not None and not isinstance(sub_layer, (ReLU, MaxPool)):
            collect_grads_recursive(sub_layer, grads_dict, f'{prefix}{attr}_')

class BasicBlock:
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        self.conv1 = Conv(out_channels, 3, in_channels, stride=stride, padding=1)
        self.bn1 = BatchNorm(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv(out_channels, 3, out_channels, stride=1, padding=1)
        self.bn2 = BatchNorm(out_channels)
        self.relu2 = ReLU()
        self.downsample = None
        self.bn_down = None
        if downsample or in_channels != out_channels:
            self.downsample = Conv(out_channels, 1, in_channels, stride=stride, padding=0)
            self.bn_down = BatchNorm(out_channels)
        self.relu_final = ReLU()

    def forward(self, X, is_inference=False):
        residual = X.copy()
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        if self.downsample:
            residual = self.downsample.forward(X)
            residual = self.bn_down.forward(residual)
        out += residual
        out = self.relu_final.forward(out)
        return out

    def backward(self, dout, is_inference=False):
        dout = self.relu_final.backward(dout)
        dout_residual = dout.copy()
        dout_main = dout.copy()
        dout_main = self.bn2.backward(dout_main)
        dout_main = self.conv2.backward(dout_main)
        dout_main = self.relu1.backward(dout_main)
        dout_main = self.bn1.backward(dout_main)
        dout_main = self.conv1.backward(dout_main)
        if self.downsample:
            dout_residual = self.bn_down.backward(dout_residual)
            dout_residual = self.downsample.backward(dout_residual)
        return dout_main + dout_residual

class ResNet18:
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        
        # Sử dụng CIFAR-style ResNet cho input nhỏ (28x28)
        # Conv 3x3 stride=1 thay vì 7x7 stride=2, bỏ MaxPool
        print("🔧 Using ResNet18-CIFAR style for 28x28 input")
        self.conv1 = Conv(64, 3, INPUT_SHAPE[0], stride=1, padding=1)
        self.bn1 = BatchNorm(64)
        self.relu = ReLU()
        self.maxpool = None  # Không dùng MaxPool cho input nhỏ
        
        self.layer1 = [BasicBlock(64, 64, 1, False) for _ in range(2)]
        self.layer2 = [BasicBlock(64, 128, 2, True)] + [BasicBlock(128, 128, 1, False) for _ in range(1)]
        self.layer3 = [BasicBlock(128, 256, 2, True)] + [BasicBlock(256, 256, 1, False) for _ in range(1)]
        self.layer4 = [BasicBlock(256, 512, 2, True)] + [BasicBlock(512, 512, 1, False) for _ in range(1)]
        self.fc = Fc(512, num_classes)
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()
        
        # All layers (bỏ maxpool)
        self.all_layers = [self.conv1, self.bn1, self.relu] + \
                          self.layer1 + self.layer2 + self.layer3 + self.layer4 + [self.fc]
        self.params = {}
        self.optimizer = Adam(lr=0.001)
        self.is_inference = False
        self._last_feature_shape = (4, 4)  # 28 -> 28 -> 14 -> 7 -> 4
        self._collect_params()

    def _collect_params(self):
        self.params = {}
        for layer in self.all_layers:
            collect_params_recursive(layer, self.params)
        print(f"✅ Đã collect {len(self.params)} params")

    def _collect_grads(self):
        grads = {}
        for layer in self.all_layers:
            collect_grads_recursive(layer, grads)
        return grads

    def forward(self, X):
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        # Bỏ maxpool cho CIFAR-style
        for layer in self.layer1 + self.layer2 + self.layer3 + self.layer4:
            out = layer.forward(out, self.is_inference)
        self._last_feature_shape = (out.shape[2], out.shape[3])
        out = xp.mean(out, axis=(2, 3))
        logits = self.fc.forward(out)
        return self.softmax.forward(logits)

    def backward(self, y_pred, y):
        dout = self.softmax.backward(y_pred, y)
        dout = self.fc.backward(dout)
        batch_size, channels = dout.shape
        H_out, W_out = self._last_feature_shape
        dout = dout.reshape(batch_size, channels, 1, 1) / (H_out * W_out)
        dout = xp.tile(dout, (1, 1, H_out, W_out))
        for layers in [self.layer4, self.layer3, self.layer2, self.layer1]:
            for layer in reversed(layers):
                dout = layer.backward(dout, self.is_inference)
        # Bỏ maxpool backward cho CIFAR-style
        dout = self.relu.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)

    def train_step(self, X, y):
        self.set_inference(False)
        # Zero gradients trước mỗi step
        self._zero_grads()
        y_pred = self.forward(X)
        loss = self.loss_fn.get(y_pred, y)
        self.backward(y_pred, y)
        grads = self._collect_grads()
        self.optimizer.update(self.params, grads)
        return loss
    
    def _zero_grads(self):
        """Reset all gradients to zero"""
        for layer in self.all_layers:
            if isinstance(layer, (Conv, Fc)):
                layer.W['grad'] = xp.zeros_like(layer.W['grad'])
                layer.b['grad'] = xp.zeros_like(layer.b['grad'])
            elif isinstance(layer, BatchNorm):
                layer.gamma['grad'] = xp.zeros_like(layer.gamma['grad'])
                layer.beta['grad'] = xp.zeros_like(layer.beta['grad'])
            # Recursive cho BasicBlock
            if hasattr(layer, 'conv1'):
                for attr in ['conv1', 'conv2', 'downsample', 'bn1', 'bn2', 'bn_down']:
                    sub_layer = getattr(layer, attr, None)
                    if sub_layer and isinstance(sub_layer, (Conv, BatchNorm)):
                        if isinstance(sub_layer, Conv):
                            sub_layer.W['grad'] = xp.zeros_like(sub_layer.W['grad'])
                            sub_layer.b['grad'] = xp.zeros_like(sub_layer.b['grad'])
                        elif isinstance(sub_layer, BatchNorm):
                            sub_layer.gamma['grad'] = xp.zeros_like(sub_layer.gamma['grad'])
                            sub_layer.beta['grad'] = xp.zeros_like(sub_layer.beta['grad'])

    def predict(self, X):
        self.set_inference(True)
        y_pred = self.forward(X)
        self.set_inference(False)
        return xp.argmax(y_pred, axis=1)

    def set_inference(self, mode=True):
        self.is_inference = mode
        for layer in self.all_layers:
            if isinstance(layer, BatchNorm):
                layer.use_moving_avg = mode
