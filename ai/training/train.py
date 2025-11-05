"""
Training Module for ResNet50
============================

Quá trình Training:
1. Forward pass: X → ResNet50 → predictions
2. Loss calculation: CrossEntropy(predictions, true_labels) 
3. Backward pass: Backpropagation để tính gradients
4. Parameter update: Adam optimizer cập nhật weights
5. Validation: Đánh giá model trên validation set

Công thức Training Loop:
For each epoch:
  For each batch:
    1. y_pred = model.forward(X_batch)
    2. loss = CrossEntropy(y_pred, y_true)
    3. gradients = model.backward(loss)
    4. optimizer.update(parameters, gradients)

Đầu vào:
- X_train: Training images (N_train, channels, height, width)
- y_train: Training labels (N_train,)
- X_val: Validation images (N_val, channels, height, width) 
- y_val: Validation labels (N_val,)
- num_classes: Số lượng classes để classify

Đầu ra:
- model: Trained ResNet50 model
"""

try:
    import cupy as np
    print("✅ Using GPU (CuPy)")
    GPU_AVAILABLE = True
except ImportError:
    import numpy as np
    print("⚠️  Using CPU (NumPy)")
    GPU_AVAILABLE = False
from ai.model.resnet import ResNet50
from ai.data.dataloader import DataLoader
from ai.configs.config import EPOCHS, LEARNING_RATE
from ai.model.layers import CrossEntropyLoss
from tqdm import tqdm
import time

def train_model(X_train, y_train, X_val, y_val, num_classes=10):  # Điều chỉnh num_classes
    from configs.config import BATCH_SIZE
    
    model = ResNet50(num_classes=num_classes)
    train_loader = DataLoader(X_train, y_train, BATCH_SIZE)
    val_loader = DataLoader(X_val, y_val, BATCH_SIZE, shuffle=False)
    
    loss_fn = CrossEntropyLoss()
    optimizer = model.optimizer  # Đã init trong model
    
    print(f"✅ Model initialized!\n")
    
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        train_loader.reset()
        total_loss = 0
        batch_count = 0
        
        # Training loop với progress bar
        num_batches = len(X_train) // BATCH_SIZE + (1 if len(X_train) % BATCH_SIZE != 0 else 0)
        pbar = tqdm(train_loader, total=num_batches, desc=f"Training", 
                    bar_format='{l_bar}{bar:30}{r_bar}')
        
        start_time = time.time()
        
        for X_batch, y_batch in pbar:
            y_onehot = train_loader.one_hot(y_batch, num_classes)
            loss = model.train_step(X_batch, y_onehot)
            total_loss += loss
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss:.4f}', 'avg_loss': f'{total_loss/batch_count:.4f}'})
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / batch_count
        
        print(f"\n📉 Training Results:")
        print(f"   - Average Loss: {avg_loss:.4f}")
        print(f"   - Time: {epoch_time:.2f}s ({epoch_time/60:.2f}m)")
        
        # Validation
        print(f"\n🔍 Validating...")
        val_acc = evaluate(model, val_loader, num_classes)
        print(f"✅ Validation Accuracy: {val_acc*100:.2f}%")
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🎯 New best validation accuracy!")
    
    print(f"\n{'='*60}")
    print(f"🏆 Training completed!")
    print(f"   - Best validation accuracy: {best_val_acc*100:.2f}%")
    print(f"{'='*60}")
    
    return model

def evaluate(model, loader, num_classes):
    """
    Evaluation Function
    ==================
    
    Đánh giá accuracy của model trên validation/test set
    
    Công thức Accuracy:
    Accuracy = Số predictions đúng / Tổng số samples
             = Σ(predicted_class == true_class) / N
    
    Đầu vào:
    - model: Trained model cần evaluate
    - loader: DataLoader chứa evaluation data
    - num_classes: Số classes (không dùng trong hàm này)
    
    Đầu ra:
    - accuracy: Float value trong khoảng [0, 1]
    
    Lưu ý:
    - Set model về inference mode (BatchNorm dùng moving stats)
    - Không cần gradients trong evaluation
    """
    correct = 0
    total = 0
    
    # Chuyển sang inference mode
    model.set_inference(True)
    
    for X_batch, y_batch in loader:
        # Forward pass để lấy predictions
        y_pred = model.predict(X_batch)  # Returns predicted class indices
        
        # Đếm số predictions đúng
        correct += np.sum(y_pred == y_batch)
        total += len(y_batch)
    
    # Chuyển về training mode
    model.set_inference(False)

    # Tính accuracy: guard against zero total to avoid ZeroDivisionError
    if total == 0:
        print("\n\u26a0\ufe0f  Warning: No samples were provided to evaluate() (total=0). Returning accuracy=0.0")
        return 0.0

    return correct / total
