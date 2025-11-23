# ai/main.py
"""
Main entry point for LeNet-5 training
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.train_lenet import train_lenet5
from ai.utils.generate_shape import generate_dataset


def main():
    """
    Main function to train LeNet-5
    """
    print("\n" + "="*70)
    print(" "*20 + "LeNet-5 Training Pipeline")
    print("="*70)
    
    # Option 1: Generate new shape dataset (if needed)
    # print("\n[Optional] Generate shape dataset...")
    # generate_dataset(num_samples_per_class=1000)
    
    # Option 2: Train LeNet-5
    print("\nStarting LeNet-5 training...")
    model = train_lenet5(
        train_folder=r'E:\PTIT\XLA\data\train',
        test_folder=r'E:\PTIT\XLA\data\test',
        n_classes=13,  # 0-9 digits + circle, square, triangle
        num_epochs=20,
        mini_batch_size=256,
        lr_global=5e-3,
        momentum=0.9,
        weight_decay=0,
        mu=0.01,
        save_dir='ai/models',
        save_interval=1
    )
    
    print("\n✅ Training complete! Model checkpoints saved in ai/models/")
    

if __name__ == "__main__":
    main()