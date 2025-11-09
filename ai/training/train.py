# ai/training/train.py
import time
from tqdm import tqdm
try:
    import cupy as cp
    xp = cp
except ImportError:
    import numpy as np
    xp = np

from ai.configs.config import EPOCHS, BATCH_SIZE, NUM_CLASSES
from ai.data.dataloader import load_dataset, DataLoader
from ai.model.resnet import ResNet18

def train_model():
    print("🔄 Đang load dữ liệu...")
    X_train, y_train = load_dataset('train')
    X_val, y_val = load_dataset('val')
    
    # Chuyển sang CuPy nếu có GPU
    if xp.__name__ == 'cupy':
        print("🚀 Chuyển data sang GPU...")
        X_train = xp.asarray(X_train)
        y_train = xp.asarray(y_train)
        X_val = xp.asarray(X_val)
        y_val = xp.asarray(y_val)
        print("✅ Data đã được chuyển sang GPU")

    train_loader = DataLoader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(X_val, y_val, BATCH_SIZE, shuffle=False)

    model = ResNet18(NUM_CLASSES)
    print(f"✅ ResNet-18 sẵn sàng! Train: {len(X_train)}, Val: {len(X_val)}")

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        if epoch % 3 == 0 and epoch > 0:
            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass

        print(f"\n{'='*60}")
        print(f"📊 Epoch {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")

        train_loader.reset()
        total_loss = 0
        batch_count = 0

        num_batches = (len(X_train) + BATCH_SIZE - 1) // BATCH_SIZE
        pbar = tqdm(range(num_batches), desc="Training")

        start_time = time.time()

        batch_idx = 0
        for X_batch, y_batch in train_loader:
            y_onehot = train_loader.one_hot(y_batch)
            if epoch == 0 and batch_idx == 0:
                print(f"🔍 Shapes: X {X_batch.shape}, y_onehot {y_onehot.shape}")
            loss = model.train_step(X_batch, y_onehot)
            total_loss += loss
            batch_count += 1
            batch_idx += 1
            pbar.set_postfix({'loss': f'{loss:.4f}', 'avg': f'{total_loss/batch_count:.4f}'})
            pbar.update(1)

        epoch_time = time.time() - start_time
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"\n📉 Train Loss: {avg_loss:.4f} | Thời gian: {epoch_time:.2f}s")

        if len(X_val) > 0:
            val_acc = evaluate(model, val_loader)
            print(f"✅ Val Acc: {val_acc*100:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"🎯 Best mới!")
                xp.save('../../best_resnet18.pth', model.params)  # Lưu model tốt nhất

    print(f"\n🏆 Hoàn tất! Best Val Acc: {best_val_acc*100:.2f}%")
    return model

def evaluate(model, loader):
    correct = total = 0
    model.set_inference(True)
    loader.reset()
    for X_batch, y_batch in loader:
        y_pred = model.predict(X_batch)
        correct += xp.sum(y_pred == y_batch)
        total += len(y_batch)
    model.set_inference(False)
    return correct / total if total > 0 else 0.0