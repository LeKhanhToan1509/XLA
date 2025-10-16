"""
ResNet (Residual Network) Implementation
======================================

Công thức chính:
- Residual connection: F(x) + x = H(x)
  Trong đó:
  - F(x): Học residual mapping
  - x: Identity shortcut connection
  - H(x): Desired underlying mapping

Đầu vào:
- Input tensor shape: (batch_size, channels, height, width)
- Labels shape: (batch_size,) hoặc (batch_size, num_classes) cho one-hot

Kiến trúc:
1. Conv layer đầu tiên: 7x7, 64 filters
2. Max pooling: 3x3
3. 4 nhóm residual blocks với số filters tăng dần: 64->128->256->512
4. Global average pooling
5. Fully connected layer cho classification
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️  Using CPU (NumPy)")
    GPU_AVAILABLE = False
from model.layers import Conv, BatchNorm, ReLU, MaxPool, Fc, Softmax, CrossEntropyLoss
from utils.optimizers import Adam
from configs.config import NUM_CLASSES, INPUT_SHAPE

class ResidualBlock:
    """
    Residual Block Implementation
    ============================
    
    Công thức Residual Block:
    y = F(x, {Wi}) + x
    
    Trong đó:
    - F(x, {Wi}): Stack của các layers (Conv -> BN -> ReLU -> Conv -> BN)
    - x: Identity shortcut (có thể có downsample nếu dimensions khác nhau)
    
    Đầu vào:
    - x: Feature maps với shape (batch_size, in_channels, height, width)
    
    Đầu ra:
    - y: Feature maps với shape (batch_size, out_channels, height/stride, width/stride)
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        self.conv1 = Conv(out_channels, 1, in_channels, stride=stride, padding=0)
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
        self.layers = [self.conv1, self.bn1, self.relu1, self.conv2, self.bn2, self.relu2, self.relu_final]
        if self.downsample:
            self.layers += [self.downsample, self.bn_down]

    def forward(self, X, is_inference=False):
        """
        Forward pass của Residual Block
        
        Công thức thực hiện:
        1. F(x) = ReLU(BN(Conv(ReLU(BN(Conv(x))))))
        2. identity = x hoặc downsample(x) nếu cần
        3. output = F(x) + identity
        4. output = ReLU(output)
        
        Đầu vào:
        - X: Input tensor (batch_size, in_channels, H, W)
        - is_inference: Boolean, chế độ inference hay training
        
        Đầu ra:
        - out: Output tensor (batch_size, out_channels, H', W')
        """
        residual = X  # Lưu identity connection
        
        # Main path: Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        
        # Shortcut connection với downsample nếu cần
        if self.downsample:
            residual = self.downsample.forward(X)
            residual = self.bn_down.forward(residual)
        
        # Element-wise addition: F(x) + x
        out += residual
        
        # Final ReLU activation
        out = self.relu_final.forward(out)
        return out

    def backward(self, dout, is_inference=False):
        dout = self.relu_final.backward(dout)
        dout_res = dout
        dout = self.bn2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        if self.downsample:
            dresidual = self.bn_down.backward(self.downsample.backward(dout_res))
            dout += self.downsample.backward(dresidual)
        else:
            dout += dout_res
        return dout

class ResNet50:
    """
    ResNet-50 Architecture Implementation
    ====================================
    
    Kiến trúc chi tiết:
    1. Conv1: 7x7, 64 filters, stride=1
    2. MaxPool: 3x3, stride=1
    3. Layer1: 3 residual blocks, 64 filters
    4. Layer2: 4 residual blocks, 128 filters, stride=2 (downsample)
    5. Layer3: 6 residual blocks, 256 filters, stride=2 (downsample)
    6. Layer4: 3 residual blocks, 512 filters, stride=2 (downsample)
    7. Global Average Pooling
    8. Fully Connected: 512 -> num_classes
    9. Softmax activation
    
    Tổng số layers: 1 + 2*(3+4+6+3) + 1 = 34 layers (conv + fc)
    
    Đầu vào:
    - X: Images tensor (batch_size, channels, height, width)
    - y: Labels tensor (batch_size,) hoặc (batch_size, num_classes)
    
    Đầu ra:
    - Logits/probabilities: (batch_size, num_classes)
    """
    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        self.conv1 = Conv(64, 7, INPUT_SHAPE[0], stride=1, padding=3)
        self.bn1 = BatchNorm(64)
        self.relu1 = ReLU()
        self.maxpool = MaxPool(3, stride=1, padding=1)  # Keep size ~24
        
        # Residual layers (scale down số blocks cho input nhỏ)
        # Layer 1: 3 blocks, 64 filters, no downsample
        self.layer1 = [ResidualBlock(64, 64, stride=1, downsample=False) for _ in range(3)]

        # Layer 2: 4 blocks, 128 filters, first block downsamples
        self.layer2 = [ResidualBlock(64, 128, stride=2, downsample=True)] + \
                    [ResidualBlock(128, 128, stride=1, downsample=False) for _ in range(3)]

        # Layer 3: 6 blocks, 256 filters, first block downsamples
        self.layer3 = [ResidualBlock(128, 256, stride=2, downsample=True)] + \
                    [ResidualBlock(256, 256, stride=1, downsample=False) for _ in range(5)]

        # Layer 4: 3 blocks, 512 filters, first block downsamples
        self.layer4 = [ResidualBlock(256, 512, stride=2, downsample=True)] + \
                    [ResidualBlock(512, 512, stride=1, downsample=False) for _ in range(2)]
        
        # Classifier
        self.avgpool = None  # Sẽ dùng np.mean trong forward
        self.fc = Fc(512 * 1 * 1, num_classes)  # Giả định sau layer4 là 1x1
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()
        
        # All layers để collect params (không bao gồm softmax vì không có params)
        self.all_layers = [self.conv1, self.bn1, self.relu1, self.maxpool] + \
                          sum([self.layer1, self.layer2, self.layer3, self.layer4], []) + \
                          [self.fc]
        
        self.params = self._collect_params()
        self.optimizer = Adam(lr=0.001)
        self.is_inference = False

    def _collect_params(self):
        params = {}
        for layer in self.all_layers:
            if isinstance(layer, (Conv, Fc)):
                params[f'W_{id(layer)}'] = layer.W['val']
                params[f'b_{id(layer)}'] = layer.b['val']
            elif isinstance(layer, BatchNorm):
                params[f'gamma_{id(layer)}'] = layer.gamma['val']
                params[f'beta_{id(layer)}'] = layer.beta['val']
        return params

    def forward(self, X):
        """
        Forward pass qua toàn bộ network
        
        Đầu vào:
        - X: Input images (batch_size, channels, height, width)
        
        Đầu ra:
        - y_pred: Predicted probabilities (batch_size, num_classes)
        """
        out = X
        
        # Conv1 + BN + ReLU + MaxPool
        out = self.conv1.forward(out)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.maxpool.forward(out)
        
        # Residual blocks
        for block in self.layer1:
            out = block.forward(out, self.is_inference)
        for block in self.layer2:
            out = block.forward(out, self.is_inference)
        for block in self.layer3:
            out = block.forward(out, self.is_inference)
        for block in self.layer4:
            out = block.forward(out, self.is_inference)
        
        # Global Average Pooling
        out = np.mean(out, axis=(2, 3))  # (batch_size, channels)
        
        # Fully Connected
        logits = self.fc.forward(out)
        
        # Softmax
        y_pred = self.softmax.forward(logits)
        
        return y_pred

    def backward(self, y_pred, y):
        """
        Backward pass để tính gradients
        
        Đầu vào:
        - y_pred: Predicted probabilities
        - y: True labels (one-hot)
        
        Lưu ý: Chỉ train FC layer, freeze feature extractor để đơn giản hóa
        """
        # Softmax + CrossEntropy backward
        dout = self.softmax.backward(y_pred, y)
        
        # FC backward - chỉ cần tính gradient cho FC layer
        dout = self.fc.backward(dout)
        
        # Không backprop qua các layer khác (frozen features)
        # Nếu muốn train full network, cần implement global avg pool backward
        # và backprop qua tất cả residual blocks

    def train_step(self, X, y):
        """
        Training step with forward + backward
        """
        # Nếu y chưa phải one-hot (shape = (batch_size,)), thì one-hot encode
        if len(y.shape) == 1:
            y_onehot = np.eye(self.num_classes)[y]
        else:
            # Nếu y đã là one-hot (shape = (batch_size, num_classes)), dùng luôn
            y_onehot = y
        
        # Set training mode
        self.set_inference(False)
        
        # Forward pass
        y_pred = self.forward(X)
        
        # Loss computation
        loss = self.loss_fn.get(y_pred, y_onehot)
        
        # Backward pass
        self.backward(y_pred, y_onehot)
        
        # Collect gradients
        grads = self._collect_grads()
        
        # Update parameters
        self.optimizer.update(self.params, grads)
        
        return loss
    
    def predict(self, X):
        self.is_inference = True
        y_pred = self.forward(X)
        self.is_inference = False
        return np.argmax(y_pred, axis=1)

    def set_inference(self, mode=True):
        self.is_inference = mode
        for layer in self.all_layers:
            if isinstance(layer, BatchNorm):
                layer.use_moving_avg = mode
    def _collect_grads(self):
        grads = {}
        for layer in self.all_layers:
            if isinstance(layer, (Conv, Fc)):
                grads[f'W_{id(layer)}'] = layer.W['grad']
                grads[f'b_{id(layer)}'] = layer.b['grad']
            elif isinstance(layer, BatchNorm):
                grads[f'gamma_{id(layer)}'] = layer.gamma['grad']
                grads[f'beta_{id(layer)}'] = layer.beta['grad']
        return grads