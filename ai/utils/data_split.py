# ai/utils/data_split.py
import os
import cv2
import numpy as np
from tqdm import tqdm
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from ai.configs.config import ROOT_DIR

def split_mnist():
    # MNIST dataset trả về PIL Image, không cần ToPILImage()
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root=ROOT_DIR, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=ROOT_DIR, train=False, download=True, transform=transform)

    # Split train: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    train_data, val_data, test_data = torch.utils.data.random_split(train_dataset, [train_size, val_size, len(train_dataset) - train_size - val_size])

    splits = {'train': train_data, 'val': val_data, 'test': test_dataset}
    for split_name, data in splits.items():
        split_dir = os.path.join(ROOT_DIR, split_name)
        for digit in range(10):
            os.makedirs(os.path.join(split_dir, str(digit)), exist_ok=True)

        for idx in tqdm(range(len(data)), desc=f"Đang lưu {split_name}"):
            img, label = data[idx]
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(split_dir, str(label), f"mnist_{idx}.png"), img_np)

    print("✅ Đã split và lưu MNIST!")