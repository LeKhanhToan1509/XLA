# LeNet-5 Architecture - Mã Giả Chi Tiết

## 📋 Tổng Quan Kiến Trúc

```
INPUT (32×32×1) 
    ↓
C1: Convolution Layer (28×28×6)
    ↓
Activation: Squashing Function
    ↓
S2: Subsampling/Pooling (14×14×6)
    ↓
C3: Convolution Layer (10×10×16)
    ↓
Activation: Squashing Function
    ↓
S4: Subsampling/Pooling (5×5×16)
    ↓
C5: Convolution Layer (1×1×120)
    ↓
Activation: Squashing Function
    ↓
F6: Fully Connected (84 neurons)
    ↓
Activation: Squashing Function
    ↓
OUTPUT: RBF Layer (13 classes)
```

---

## 1️⃣ Lớp C1 - Convolution Layer

### Mã Giả Forward Propagation

```pseudocode
FUNCTION C1_ForwardProp(input_image):
    INPUT:
        input_image: tensor (batch_size, 32, 32, 1)
        
    PARAMETERS:
        weight: tensor (5, 5, 1, 6)  // 6 filters, size 5×5
        bias: tensor (1, 1, 1, 6)    // 6 bias values
        stride: 1
        padding: 0
        
    PROCESS:
        output_height = (32 - 5 + 2*0)/1 + 1 = 28
        output_width = (32 - 5 + 2*0)/1 + 1 = 28
        
        FOR each sample in batch:
            FOR each filter k from 0 to 5:
                FOR i from 0 to 27:
                    FOR j from 0 to 27:
                        // Extract 5×5 region
                        region = input_image[i:i+5, j:j+5, :]
                        
                        // Convolution operation
                        output[i, j, k] = SUM(region * weight[:,:,:,k]) + bias[k]
                        
    OUTPUT:
        output_map: tensor (batch_size, 28, 28, 6)
        
    RETURN output_map
END FUNCTION
```

### Mã Giả Backpropagation

```pseudocode
FUNCTION C1_BackProp(dZ):
    INPUT:
        dZ: gradient from next layer (batch_size, 28, 28, 6)
        
    PROCESS:
        // Gradient w.r.t. weights
        dW = ZEROS(5, 5, 1, 6)
        FOR each sample in batch:
            FOR each filter k:
                FOR i, j in output positions:
                    region = input[i:i+5, j:j+5, :]
                    dW[:,:,:,k] += region * dZ[i, j, k]
        dW = dW / batch_size
        
        // Gradient w.r.t. bias
        db = SUM(dZ, axis=(0,1,2)) / batch_size
        
        // Gradient w.r.t. input
        dA_prev = ZEROS_LIKE(input_image)
        FOR each filter k:
            FOR i, j in output positions:
                dA_prev[i:i+5, j:j+5, :] += weight[:,:,:,k] * dZ[i,j,k]
                
    RETURN dA_prev, dW, db
END FUNCTION
```

### Mã Giả Weight Update

```pseudocode
FUNCTION Update_Weights(weight, bias, dW, db, v_w, v_b, lr, momentum, weight_decay):
    // Momentum update
    v_w = momentum * v_w - weight_decay * lr * weight - lr * dW
    v_b = momentum * v_b - weight_decay * lr * bias - lr * db
    
    // Update parameters
    weight = weight + v_w
    bias = bias + v_b
    
    RETURN weight, bias, v_w, v_b
END FUNCTION
```

---

## 2️⃣ Activation Function - Squashing Function

### Mã Giả

```pseudocode
    FUNCTION LeNet5_Squash(x):
        // Original LeNet-5 activation: f(x) = A * tanh(S * x)
        // Typically A = 1.7159, S = 2/3
        
        INPUT:
            x: input tensor
            
        PARAMETERS:
            A = 1.7159
            S = 0.6666667
            
        FORWARD:
            output = A * tanh(S * x)
            
        DERIVATIVE:
            // f'(x) = A * S * (1 - tanh²(S*x))
            tanh_val = tanh(S * x)
            derivative = A * S * (1 - tanh_val²)
            
        RETURN output, derivative
    END FUNCTION
```

### Alternative: Sigmoid

```pseudocode
FUNCTION Sigmoid(x):
    FORWARD:
        output = 1 / (1 + exp(-x))
        
    DERIVATIVE:
        derivative = output * (1 - output)
        
    RETURN output, derivative
END FUNCTION
```

---

## 3️⃣ Lớp S2 - Subsampling/Pooling Layer

### Mã Giả Average Pooling

```pseudocode
FUNCTION S2_AveragePooling(input_map):
    INPUT:
        input_map: tensor (batch_size, 28, 28, 6)
        
    PARAMETERS:
        pool_size: 2×2
        stride: 2
        
    PROCESS:
        output_height = (28 - 2) / 2 + 1 = 14
        output_width = (28 - 2) / 2 + 1 = 14
        
        FOR each sample in batch:
            FOR each channel k from 0 to 5:
                FOR i from 0 to 13:
                    FOR j from 0 to 13:
                        // Extract 2×2 region
                        region = input_map[i*2:i*2+2, j*2:j*2+2, k]
                        
                        // Average pooling
                        output[i, j, k] = MEAN(region)
                        
    OUTPUT:
        output_map: tensor (batch_size, 14, 14, 6)
        
    RETURN output_map
END FUNCTION
```

### Mã Giả Backpropagation

```pseudocode
FUNCTION S2_BackProp(dA):
    INPUT:
        dA: gradient from next layer (batch_size, 14, 14, 6)
        
    PROCESS:
        dA_prev = ZEROS(batch_size, 28, 28, 6)
        
        FOR each position (i, j, k) in dA:
            // Distribute gradient equally to 2×2 region
            gradient_per_pixel = dA[i, j, k] / 4
            
            FOR m from 0 to 1:
                FOR n from 0 to 1:
                    dA_prev[i*2+m, j*2+n, k] = gradient_per_pixel
                    
    RETURN dA_prev
END FUNCTION
```

---

## 4️⃣ Lớp C3 - Convolution với Mapping Đặc Biệt

### Mã Giả Forward Propagation

```pseudocode
FUNCTION C3_ForwardProp(input_map):
    INPUT:
        input_map: tensor (batch_size, 14, 14, 6)
        
    PARAMETERS:
        // C3 mapping: each of 16 output maps connects to subset of input maps
        mapping = [
            [0,1,2], [1,2,3], [2,3,4], [3,4,5], [4,5,0], [5,0,1],  // 6 maps: 3 inputs
            [0,1,2,3], [1,2,3,4], [2,3,4,5],                        // 6 maps: 4 inputs
            [3,4,5,0], [4,5,0,1], [5,0,1,2],                        // 3 maps: 4 inputs
            [0,1,3,4], [1,2,4,5], [0,2,3,5],                        // 3 maps: 4 inputs
            [0,1,2,3,4,5]                                           // 1 map: 6 inputs
        ]
        
        FOR each output map i from 0 to 15:
            weight[i]: tensor (5, 5, len(mapping[i]), 1)
            bias[i]: scalar
            
    PROCESS:
        output = EMPTY(batch_size, 10, 10, 16)
        
        FOR each output map i from 0 to 15:
            // Select input channels according to mapping
            selected_inputs = input_map[:, :, :, mapping[i]]
            
            // Perform convolution
            output[:, :, :, i] = CONVOLVE(selected_inputs, weight[i]) + bias[i]
            
    OUTPUT:
        output_map: tensor (batch_size, 10, 10, 16)
        
    RETURN output_map
END FUNCTION
```

### Lý Do Sử Dụng Mapping

```
🎯 MỤC ĐÍCH:
1. Giảm số lượng tham số (connections)
2. Phá vỡ tính đối xứng giữa các feature maps
3. Buộc mỗi feature map học các features khác nhau
4. Tổng connections: 1516 instead of 2400 (nếu fully connected)

📊 PHÂN BỐ CONNECTIONS:
- 6 maps kết nối với 3 input channels
- 6 maps kết nối với 4 input channels  
- 3 maps kết nối với 4 input channels
- 1 map kết nối với tất cả 6 input channels
```

---

## 5️⃣ Lớp S4 - Subsampling Layer (Tương tự S2)

```pseudocode
FUNCTION S4_AveragePooling(input_map):
    // Giống S2, nhưng với input (10, 10, 16)
    INPUT:
        input_map: tensor (batch_size, 10, 10, 16)
        
    OUTPUT:
        output_map: tensor (batch_size, 5, 5, 16)
        
    // Logic tương tự S2_AveragePooling
END FUNCTION
```

---

## 6️⃣ Lớp C5 - Convolution Layer (Full Connection)

### Mã Giả

```pseudocode
FUNCTION C5_ForwardProp(input_map):
    INPUT:
        input_map: tensor (batch_size, 5, 5, 16)
        
    PARAMETERS:
        weight: tensor (5, 5, 16, 120)  // 120 filters
        bias: tensor (1, 1, 1, 120)
        
    PROCESS:
        // Convolution with 5×5 kernel on 5×5 input
        // Output size: (5-5+0)/1 + 1 = 1
        
        FOR each sample in batch:
            FOR each filter k from 0 to 119:
                output[0, 0, k] = SUM(input_map * weight[:,:,:,k]) + bias[k]
                
    OUTPUT:
        output_map: tensor (batch_size, 1, 1, 120)
        
    NOTE:
        // C5 hoạt động như Fully Connected layer
        // Mỗi output neuron kết nối với toàn bộ 5×5×16 = 400 inputs
        
    RETURN output_map
END FUNCTION
```

---

## 7️⃣ Lớp F6 - Fully Connected Layer

### Mã Giả Forward Propagation

```pseudocode
FUNCTION F6_ForwardProp(input_array):
    INPUT:
        input_array: tensor (batch_size, 120)
        
    PARAMETERS:
        weight: matrix (120, 84)
        bias: vector (84,)
        
    PROCESS:
        // Matrix multiplication
        output = input_array @ weight + bias
        // @ là ký hiệu matrix multiplication
        
        // Explicitly:
        FOR each sample i in batch:
            FOR each output neuron j from 0 to 83:
                output[i, j] = SUM(input_array[i, :] * weight[:, j]) + bias[j]
                
    OUTPUT:
        output: tensor (batch_size, 84)
        
    RETURN output
END FUNCTION
```

### Mã Giả Backpropagation

```pseudocode
FUNCTION F6_BackProp(dZ):
    INPUT:
        dZ: gradient from next layer (batch_size, 84)
        
    PROCESS:
        // Gradient w.r.t. input
        dA = dZ @ weight.T
        // dA shape: (batch_size, 120)
        
        // Gradient w.r.t. weight
        dW = input_array.T @ dZ / batch_size
        // dW shape: (120, 84)
        
        // Gradient w.r.t. bias
        db = SUM(dZ, axis=0) / batch_size
        // db shape: (84,)
        
    RETURN dA, dW, db
END FUNCTION
```

---

## 8️⃣ Lớp OUTPUT - RBF (Radial Basis Function) Layer

### Mã Giả Forward Propagation

```pseudocode
FUNCTION RBF_ForwardProp(input_array, label, mode):
    INPUT:
        input_array: tensor (batch_size, 84)
        label: tensor (batch_size,) - ground truth labels
        mode: 'train' hoặc 'test'
        
    PARAMETERS:
        weight: matrix (n_classes, 84)
        // Mỗi class có 1 prototype vector 84 chiều
        // Thường khởi tạo bằng bitmap patterns của chữ số
        
    PROCESS - TRAINING MODE:
        IF mode == 'train':
            loss = 0
            FOR each sample i in batch:
                // Lấy prototype vector của class đúng
                prototype = weight[label[i], :]
                
                // Euclidean distance
                distance = input_array[i, :] - prototype
                
                // RBF loss (squared Euclidean distance)
                sample_loss = 0.5 * SUM(distance²)
                loss += sample_loss
                
            RETURN loss
            
    PROCESS - TESTING MODE:
        IF mode == 'test':
            predictions = EMPTY(batch_size)
            
            FOR each sample i in batch:
                distances = EMPTY(n_classes)
                
                // Tính khoảng cách đến mỗi prototype
                FOR each class c from 0 to n_classes-1:
                    prototype = weight[c, :]
                    distances[c] = SUM((input_array[i, :] - prototype)²)
                
                // Chọn class có khoảng cách nhỏ nhất
                predictions[i] = ARGMIN(distances)
            
            // Tính error rate
            error_count = COUNT(predictions != label)
            
            RETURN error_count, predictions
END FUNCTION
```

### Mã Giả Backpropagation

```pseudocode
FUNCTION RBF_BackProp(label):
    INPUT:
        label: tensor (batch_size,) - ground truth labels
        
    PROCESS:
        dy_predict = EMPTY(batch_size, 84)
        
        FOR each sample i in batch:
            // Gradient of loss w.r.t. input
            prototype = weight[label[i], :]
            dy_predict[i, :] = input_array[i, :] - prototype
            
    OUTPUT:
        dy_predict: tensor (batch_size, 84)
        
    RETURN dy_predict
END FUNCTION
```

### Khởi Tạo RBF Weights

```pseudocode
FUNCTION Initialize_RBF_Weights(n_classes):
    // Tạo bitmap patterns cho mỗi class
    
    IF n_classes == 10:  // MNIST digits
        // Sử dụng 7×12 bitmap cho mỗi chữ số
        bitmap = CREATE_DIGIT_BITMAPS()  // 10×84 matrix
        
    ELSE:  // Custom classes
        // Sử dụng random initialization hoặc custom patterns
        bitmap = RANDOM_NORMAL(n_classes, 84, mean=0, std=0.1)
        
    RETURN bitmap
END FUNCTION

FUNCTION CREATE_DIGIT_BITMAPS():
    // Ví dụ bitmap cho số "0"
    bitmap_0 = [
        [1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1],
        [1,1,1,1,1,1,1]
    ]
    // Flatten to 84 dimensions: 7×12 = 84
    
    // Tương tự cho các chữ số khác (1-9)
    
    RETURN all_bitmaps  // Shape: (10, 84)
END FUNCTION
```

---

## 9️⃣ SDLM - Stochastic Diagonal Levenberg-Marquardt

### Mã Giả Adaptive Learning Rate

```pseudocode
FUNCTION SDLM_UpdateLearningRate(layer, d2Z, mu, lr_global):
    PURPOSE:
        // Tự động điều chỉnh learning rate cho từng layer
        // dựa trên second-order derivative information
        
    INPUT:
        d2Z: second derivative of loss w.r.t. layer output
        mu: regularization parameter (e.g., 0.01)
        lr_global: global learning rate (e.g., 5e-3)
        
    PROCESS:
        IF layer is ConvLayer:
            // Compute second derivative w.r.t. weights
            d2W = COMPUTE_CONV_SECOND_DERIVATIVE(d2Z)
            
        ELSE IF layer is FCLayer:
            // Compute second derivative w.r.t. weights
            d2W = input.T² @ d2Z
            
        // Diagonal approximation of Hessian
        h = SUM(d2W) / batch_size
        
        // Adaptive learning rate (Levenberg-Marquardt style)
        layer.lr = lr_global / (mu + h)
        
    NOTE:
        // Nếu h lớn (curvature cao) → lr nhỏ (careful steps)
        // Nếu h nhỏ (curvature thấp) → lr lớn (aggressive steps)
        
    RETURN d2A_prev  // Backpropagate second derivative
END FUNCTION
```

---

## 🔟 Complete Forward & Backward Pass

### Mã Giả Forward Propagation Đầy Đủ

```pseudocode
FUNCTION LeNet5_Forward(input_image, label, mode):
    INPUT:
        input_image: tensor (batch_size, 32, 32, 1)
        label: tensor (batch_size,)
        mode: 'train' hoặc 'test'
        
    FORWARD PASS:
        // Layer 1: C1 + Activation + S2
        C1_out = C1.forward(input_image)        // (batch, 28, 28, 6)
        a1_out = Squash(C1_out)                 // (batch, 28, 28, 6)
        S2_out = S2.forward(a1_out)             // (batch, 14, 14, 6)
        
        // Layer 2: C3 + Activation + S4
        C3_out = C3.forward(S2_out)             // (batch, 10, 10, 16)
        a2_out = Squash(C3_out)                 // (batch, 10, 10, 16)
        S4_out = S4.forward(a2_out)             // (batch, 5, 5, 16)
        
        // Layer 3: C5 + Activation
        C5_out = C5.forward(S4_out)             // (batch, 1, 1, 120)
        a3_out = Squash(C5_out)                 // (batch, 1, 1, 120)
        
        // Flatten
        flatten = RESHAPE(a3_out, (batch, 120)) // (batch, 120)
        
        // Layer 4: F6 + Activation
        F6_out = F6.forward(flatten)            // (batch, 84)
        a4_out = Squash(F6_out)                 // (batch, 84)
        
        // Output: RBF Layer
        output = RBF.forward(a4_out, label, mode)
        
        IF mode == 'train':
            RETURN total_loss  // Sum of RBF losses
        ELSE:
            RETURN error_count, predictions
END FUNCTION
```

### Mã Giả Backward Propagation Đầy Đủ

```pseudocode
FUNCTION LeNet5_Backward(momentum, weight_decay):
    INPUT:
        momentum: momentum coefficient (e.g., 0.9)
        weight_decay: L2 regularization (e.g., 0 or 5e-4)
        
    BACKWARD PASS:
        // Output layer
        dy_pred = RBF.backward()                // (batch, 84)
        
        // Layer 4: F6 backward
        dy_pred = Squash.backward(dy_pred)      // (batch, 84)
        F6_grad = F6.backward(dy_pred, momentum, weight_decay)
        F6_grad = RESHAPE(F6_grad, (batch, 1, 1, 120))
        
        // Layer 3: C5 backward
        C5_grad = Squash.backward(F6_grad)      // (batch, 1, 1, 120)
        C5_grad = C5.backward(C5_grad, momentum, weight_decay)
        
        // Layer 2: S4, C3 backward
        S4_grad = S4.backward(C5_grad)          // (batch, 5, 5, 16)
        S4_grad = Squash.backward(S4_grad)      // (batch, 5, 5, 16)
        C3_grad = C3.backward(S4_grad, momentum, weight_decay)
        
        // Layer 1: S2, C1 backward
        S2_grad = S2.backward(C3_grad)          // (batch, 10, 10, 16)
        S2_grad = Squash.backward(S2_grad)      // (batch, 10, 10, 16)
        C1_grad = C1.backward(S2_grad, momentum, weight_decay)
        
    NOTE:
        // Weights đã được update trong mỗi layer.backward()
        // Sử dụng momentum và weight_decay
END FUNCTION
```

---

## 1️⃣1️⃣ Training Loop

### Mã Giả Training Algorithm

```pseudocode
FUNCTION Train_LeNet5(train_data, test_data, hyperparameters):
    INPUT:
        train_data: training dataset
        test_data: test dataset
        hyperparameters: {
            num_epochs: 20,
            batch_size: 256,
            lr_global: 5e-3,
            momentum: 0.9,
            weight_decay: 0,
            mu: 0.01  // for SDLM
        }
        
    INITIALIZATION:
        model = LeNet5(n_classes=10)
        
    TRAINING LOOP:
        FOR epoch from 1 to num_epochs:
            
            // Shuffle training data
            SHUFFLE(train_data)
            
            // Mini-batch training
            num_batches = CEIL(len(train_data) / batch_size)
            epoch_loss = 0
            
            FOR batch_idx from 0 to num_batches-1:
                // Get mini-batch
                start_idx = batch_idx * batch_size
                end_idx = MIN(start_idx + batch_size, len(train_data))
                batch_images = train_data[start_idx:end_idx].images
                batch_labels = train_data[start_idx:end_idx].labels
                
                // Preprocess (normalize, zero-pad)
                batch_images = PREPROCESS(batch_images)
                
                // ===== Forward Pass =====
                loss = model.Forward(batch_images, batch_labels, mode='train')
                epoch_loss += loss
                
                // ===== Backward Pass =====
                model.Backward(momentum, weight_decay)
                
                // ===== SDLM: Update Learning Rates =====
                model.SDLM(mu, lr_global)
                
                // Print progress
                IF batch_idx % 10 == 0:
                    PRINT("Epoch {}/{}, Batch {}/{}, Loss: {:.4f}".format(
                        epoch, num_epochs, batch_idx, num_batches, loss
                    ))
            
            // ===== Validation =====
            train_error = EVALUATE(model, train_data)
            test_error = EVALUATE(model, test_data)
            
            PRINT("Epoch {}: Train Error = {:.2f}%, Test Error = {:.2f}%".format(
                epoch, train_error*100, test_error*100
            ))
            
            // ===== Save Checkpoint =====
            IF epoch % 5 == 0:
                SAVE_MODEL(model, "model_epoch_{}.pkl".format(epoch))
        
        RETURN model
END FUNCTION
```

### Mã Giả Evaluation

```pseudocode
FUNCTION EVALUATE(model, dataset):
    total_error = 0
    total_samples = 0
    
    FOR batch in dataset:
        batch_images = PREPROCESS(batch.images)
        batch_labels = batch.labels
        
        // Forward pass in test mode
        error_count, predictions = model.Forward(
            batch_images, batch_labels, mode='test'
        )
        
        total_error += error_count
        total_samples += len(batch_labels)
    
    error_rate = total_error / total_samples
    
    RETURN error_rate
END FUNCTION
```

---

## 1️⃣2️⃣ Preprocessing

### Mã Giả Preprocessing Pipeline

```pseudocode
FUNCTION PREPROCESS(images, method='lenet5'):
    INPUT:
        images: tensor (batch_size, height, width, channels)
        method: 'lenet5', 'mnist', hoặc 'custom'
        
    PROCESS:
        IF method == 'lenet5':
            // 1. Zero padding: 28×28 → 32×32
            images = ZERO_PAD(images, pad_size=2)
            
            // 2. Normalize to [-0.1, 1.175]
            // Formula: x_norm = (x/255 - mean) / std
            mean = 0.1307
            std = 0.3081
            images = (images / 255.0 - mean) / std
            
        ELSE IF method == 'mnist':
            // Simple normalization: [0, 255] → [0, 1]
            images = images / 255.0
            
        ELSE IF method == 'custom':
            // Custom preprocessing
            images = CUSTOM_NORMALIZE(images)
    
    OUTPUT:
        preprocessed_images: tensor (batch_size, 32, 32, channels)
        
    RETURN preprocessed_images
END FUNCTION

FUNCTION ZERO_PAD(images, pad_size):
    // Add zero padding around images
    padded = ZEROS(
        batch_size, 
        height + 2*pad_size, 
        width + 2*pad_size, 
        channels
    )
    
    padded[:, pad_size:-pad_size, pad_size:-pad_size, :] = images
    
    RETURN padded
END FUNCTION
```

---

## 1️⃣3️⃣ Hyperparameters Summary

```pseudocode
═══════════════════════════════════════════════════
                HYPERPARAMETERS
═══════════════════════════════════════════════════

ARCHITECTURE:
    Input Size:           32 × 32 × 1
    
    C1 Layer:
        Filters:          6
        Kernel Size:      5 × 5
        Stride:           1
        Padding:          0
        Output:           28 × 28 × 6
        Parameters:       (5*5*1+1)*6 = 156
    
    S2 Layer (Pooling):
        Pool Size:        2 × 2
        Stride:           2
        Output:           14 × 14 × 6
    
    C3 Layer:
        Filters:          16
        Kernel Size:      5 × 5
        Connections:      Selected (not fully connected)
        Output:           10 × 10 × 16
        Parameters:       ~1,516
    
    S4 Layer (Pooling):
        Pool Size:        2 × 2
        Stride:           2
        Output:           5 × 5 × 16
    
    C5 Layer:
        Filters:          120
        Kernel Size:      5 × 5
        Output:           1 × 1 × 120
        Parameters:       (5*5*16+1)*120 = 48,120
    
    F6 Layer:
        Input:            120
        Output:           84
        Parameters:       120*84 + 84 = 10,164
    
    RBF Output:
        Input:            84
        Output:           10 classes
        Parameters:       10*84 = 840

TOTAL PARAMETERS:         ~60,000

TRAINING HYPERPARAMETERS:
    Epochs:               20
    Batch Size:           256
    Learning Rate:        5e-3 (adaptive via SDLM)
    Momentum:             0.9
    Weight Decay:         0 (no L2 regularization)
    SDLM mu:              0.01
    
OPTIMIZATION:
    Method:               SGD with Momentum + SDLM
    Weight Init:          Gaussian distribution
    Activation:           LeNet5 Squashing Function
    Loss:                 RBF (Euclidean Distance)
```

---

## 1️⃣4️⃣ Complexity Analysis

```pseudocode
═══════════════════════════════════════════════════
            COMPUTATIONAL COMPLEXITY
═══════════════════════════════════════════════════

FORWARD PROPAGATION:

C1: Convolution
    Operations: 28 × 28 × 6 × (5×5×1) = 117,600
    
S2: Pooling
    Operations: 14 × 14 × 6 × 4 = 4,704
    
C3: Convolution
    Operations: 10 × 10 × 16 × (5×5×avg_connections) ≈ 240,000
    
S4: Pooling
    Operations: 5 × 5 × 16 × 4 = 1,600
    
C5: Convolution
    Operations: 1 × 1 × 120 × (5×5×16) = 48,000
    
F6: Fully Connected
    Operations: 120 × 84 = 10,080
    
RBF: Distance Computation
    Operations: 84 × 10 = 840

TOTAL FORWARD: ~422,824 operations per sample

BACKWARD PROPAGATION:
    Approximately 2-3× forward pass complexity

MEMORY USAGE:
    Input:               32 × 32 × 1 = 1,024 bytes
    C1 output:           28 × 28 × 6 = 4,704 bytes
    S2 output:           14 × 14 × 6 = 1,176 bytes
    C3 output:           10 × 10 × 16 = 1,600 bytes
    S4 output:           5 × 5 × 16 = 400 bytes
    C5 output:           120 bytes
    F6 output:           84 bytes
    
    Weights:             ~60,000 parameters × 4 bytes = 240 KB
    
BATCH PROCESSING (batch_size = 256):
    Forward:             ~108 million operations
    Backward:            ~200-300 million operations
```

---

## 📚 References & Notes

```
🔗 PAPER: "Gradient-Based Learning Applied to Document Recognition"
   Authors: Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner
   Year: 1998
   
📌 KEY INNOVATIONS:
   1. Convolutional layers for translation invariance
   2. Subsampling for dimensionality reduction
   3. Sparse connections in C3 for feature diversity
   4. RBF output layer with Euclidean distance
   5. SDLM for adaptive learning rates
   
🎯 APPLICATIONS:
   - Handwritten digit recognition (MNIST)
   - Document analysis
   - Character recognition
   - Pattern recognition
   
💡 MODERN VARIATIONS:
   - Replace subsampling with max pooling
   - Use ReLU instead of tanh
   - Add batch normalization
   - Replace RBF with softmax
   - Add dropout for regularization
```

---

**📝 Lưu ý khi sử dụng mã giả này:**

1. ✅ Các công thức toán học có thể cần điều chỉnh cho framework cụ thể
2. ✅ Indexing có thể khác nhau giữa các ngôn ngữ lập trình
3. ✅ Implementation thực tế cần xử lý edge cases
4. ✅ Performance optimization (vectorization, GPU) không được thể hiện
5. ✅ Error handling và validation không được bao gồm

**🚀 Để implement:**
- Sử dụng NumPy cho Python
- Sử dụng PyTorch/TensorFlow cho deep learning frameworks
- Tối ưu hóa với CUDA cho GPU acceleration
