"""
LeNet-5 Training Module
Train LeNet-5 CNN for 13-class classification (MNIST digits + shapes)
"""

import numpy as np
import pickle
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.utils.LayerObjects import LeNet5
from ai.utils.utils_func import readDatasetFromFolder, zero_pad, normalize, random_mini_batches


def train_lenet5(
    train_folder=r'E:\PTIT\XLA\data\train',
    test_folder=r'E:\PTIT\XLA\data\test',
    n_classes=13,
    num_epochs=20,
    mini_batch_size=256,
    lr_global=5e-3,
    momentum=0.9,
    weight_decay=0,
    mu=0.01,
    save_dir='ai/models',
    save_interval=1
):
    """
    Train LeNet-5 model
    
    Args:
        train_folder: Path to training data folder
        test_folder: Path to test data folder
        n_classes: Number of classes (10 for MNIST, 13 for MNIST+shapes)
        num_epochs: Number of training epochs
        mini_batch_size: Mini-batch size for training
        lr_global: Global learning rate for SDLM
        momentum: Momentum for SGD
        weight_decay: L2 regularization parameter
        mu: SDLM parameter
        save_dir: Directory to save model checkpoints
        save_interval: Save model every N epochs
        
    Returns:
        ConvNet: Trained LeNet-5 model
    """
    
    print("="*70)
    print("LeNet-5 Training - MNIST + Shapes Classification")
    print("="*70)
    
    # ========== 1. Load Dataset ==========
    print("\n[1/5] Loading dataset...")
    (train_image, train_label) = readDatasetFromFolder(train_folder)
    (test_image, test_label) = readDatasetFromFolder(test_folder)
    
    n_m, n_m_test = len(train_label), len(test_label)
    print(f"\n✓ Dataset Info:")
    print(f"  - Training samples: {n_m}")
    print(f"  - Test samples: {n_m_test}")
    print(f"  - Image shape: {train_image.shape[1:]}")
    print(f"  - Unique labels: {np.unique(train_label)}")
    print(f"  - Number of classes: {n_classes}")
    
    # ========== 2. Preprocessing ==========
    print("\n[2/5] Preprocessing images...")
    train_image_normalized_pad = normalize(zero_pad(train_image[:,:,:,np.newaxis], 2), 'lenet5')
    test_image_normalized_pad = normalize(zero_pad(test_image[:,:,:,np.newaxis], 2), 'lenet5')
    
    print(f"✓ After padding (pad=2):")
    print(f"  - Training shape: {train_image_normalized_pad.shape}")
    print(f"  - Test shape: {test_image_normalized_pad.shape}")
    
    # ========== 3. Initialize Model ==========
    print("\n[3/5] Initializing LeNet-5 model...")
    
    # C3 mapping configuration (from original paper)
    C3_mapping = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],
                  [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],
                  [0,1,3,4],[1,2,4,5],[0,2,3,5],
                  [0,1,2,3,4,5]]
    
    ConvNet = LeNet5(n_classes=n_classes, C3_mapping=C3_mapping)
    
    print("✓ Model architecture:")
    print("  C1 (Conv 5x5x1x6) → S2 (AvgPool 2x2) → C3 (Conv 5x5x6x16) →")
    print("  S4 (AvgPool 2x2) → C5 (Conv 5x5x16x120) → F6 (FC 120→84) →")
    print(f"  RBF (Output {n_classes} classes)")
    
    # ========== 4. Training Loop ==========
    print("\n[4/5] Training model...")
    print(f"Hyperparameters:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {mini_batch_size}")
    print(f"  - Global LR: {lr_global}")
    print(f"  - Momentum: {momentum}")
    print(f"  - Weight decay: {weight_decay}")
    print(f"  - SDLM mu: {mu}")
    print("-"*70)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    train_loss_history = []
    test_acc_history = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Shuffle and create mini-batches
        minibatches = random_mini_batches(train_image_normalized_pad, train_label, mini_batch_size)
        
        epoch_loss = 0
        num_batches = len(minibatches)
        
        # Training loop
        for batch_idx, (minibatch_X, minibatch_Y) in enumerate(minibatches):
            # Forward propagation
            loss = ConvNet.Forward_Propagation(minibatch_X, minibatch_Y, mode='train')
            epoch_loss += loss
            
            # Backward propagation
            ConvNet.Back_Propagation(momentum, weight_decay)
            
            # SDLM for learning rate adjustment
            ConvNet.SDLM(mu, lr_global)
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{num_epochs} - Batch {batch_idx+1}/{num_batches} - Loss: {avg_loss:.4f}", end='\r')
        
        # Calculate average loss
        avg_epoch_loss = epoch_loss / num_batches
        train_loss_history.append(avg_epoch_loss)
        
        # Test accuracy
        test_acc = evaluate_model(ConvNet, test_image_normalized_pad, test_label, mini_batch_size)
        test_acc_history.append(test_acc)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_epoch_loss:.4f} - Test Acc: {test_acc:.2f}% - Time: {epoch_time:.1f}s")
        
        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            save_path = os.path.join(save_dir, f'model_weights_{epoch}.pkl')
            save_model(ConvNet, save_path, epoch)
            print(f"  → Model saved: {save_path}")
    
    # ========== 5. Final Evaluation ==========
    print("\n[5/5] Final evaluation...")
    final_test_acc = evaluate_model(ConvNet, test_image_normalized_pad, test_label, mini_batch_size)
    final_train_acc = evaluate_model(ConvNet, train_image_normalized_pad, train_label, mini_batch_size)
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")
    print(f"\nBest Test Accuracy: {max(test_acc_history):.2f}% (Epoch {np.argmax(test_acc_history)+1})")
    
    # Save final model
    final_save_path = os.path.join(save_dir, 'model_weights_final.pkl')
    save_model(ConvNet, final_save_path, num_epochs-1)
    print(f"\nFinal model saved: {final_save_path}")
    
    return ConvNet


def evaluate_model(model, images, labels, batch_size=256):
    """
    Evaluate model accuracy on given dataset
    
    Args:
        model: LeNet5 model
        images: Input images (n_samples, 32, 32, 1)
        labels: Ground truth labels (n_samples,)
        batch_size: Batch size for evaluation
        
    Returns:
        accuracy: Accuracy percentage
    """
    n_samples = len(labels)
    n_batches = int(np.ceil(n_samples / batch_size))
    correct = 0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        
        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Forward propagation in test mode
        _, predictions = model.Forward_Propagation(batch_images, batch_labels, mode='test')
        
        # Count correct predictions
        correct += np.sum(predictions == batch_labels)
    
    accuracy = (correct / n_samples) * 100
    return accuracy


def save_model(model, save_path, epoch):
    """
    Save model weights to pickle file
    
    Args:
        model: LeNet5 model
        save_path: Path to save pickle file
        epoch: Current epoch number
    """
    model_state = {
        'C1_weight': model.C1.weight,
        'C1_bias': model.C1.bias,
        'C3_wb': model.C3.wb,
        'C5_weight': model.C5.weight,
        'C5_bias': model.C5.bias,
        'F6_weight': model.F6.weight,
        'F6_bias': model.F6.bias,
        'RBF_weight': model.RBF.weight,
        'epoch': epoch
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_state, f)


def load_model(load_path, n_classes=13):
    """
    Load model weights from pickle file
    
    Args:
        load_path: Path to pickle file
        n_classes: Number of classes
        
    Returns:
        model: LeNet5 model with loaded weights
    """
    print(f"Loading model from {load_path}...")
    
    with open(load_path, 'rb') as f:
        model_state = pickle.load(f)
    
    # Initialize new model
    C3_mapping = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],
                  [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],
                  [0,1,3,4],[1,2,4,5],[0,2,3,5],
                  [0,1,2,3,4,5]]
    
    model = LeNet5(n_classes=n_classes, C3_mapping=C3_mapping)
    
    # Restore weights
    model.C1.weight = model_state['C1_weight']
    model.C1.bias = model_state['C1_bias']
    model.C3.wb = model_state['C3_wb']
    model.C5.weight = model_state['C5_weight']
    model.C5.bias = model_state['C5_bias']
    model.F6.weight = model_state['F6_weight']
    model.F6.bias = model_state['F6_bias']
    model.RBF.weight = model_state['RBF_weight']
    
    print(f"✓ Model loaded from epoch {model_state['epoch']}")
    
    return model
