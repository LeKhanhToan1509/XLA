import cv2
import numpy as np
import os
import random

# Cấu hình đường dẫn
ROOT_DIR = r'E:\PTIT\XLA\data'

def create_clean_background(size=28):
    """Tạo background đơn giản với màu xám đồng nhất"""
    bg_color = random.randint(20, 80)  # Dark background
    return np.full((size, size), bg_color, dtype=np.uint8)

def generate_circle(size=28, save_path=None):
    """Generate circle image với background sạch và hình tròn ở giữa"""
    img = create_clean_background(size)
    
    # Vẽ circle ở chính giữa
    center = (size // 2, size // 2)
    radius = random.randint(6, 10)  # Larger radius than before
    color = random.randint(200, 255)  # White-ish color
    thickness = -1  # Filled circle
    
    cv2.circle(img, center, radius, color, thickness)
    
    # Light augmentation
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(save_path, img)
    return img

def generate_square(size=28, save_path=None):
    """Generate square image với background sạch và hình vuông ở giữa"""
    img = create_clean_background(size)
    
    # Vẽ square ở chính giữa
    s = random.randint(6, 10)  # Side length
    center = size // 2
    top_left = (center - s, center - s)
    bottom_right = (center + s, center + s)
    color = random.randint(200, 255)  # White-ish color
    
    cv2.rectangle(img, top_left, bottom_right, color, -1)
    
    # Light augmentation
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(save_path, img)
    return img

def generate_triangle(size=28, save_path=None):
    """Generate triangle image với background sạch và tam giác ở giữa"""
    img = create_clean_background(size)
    
    # Vẽ triangle ở chính giữa
    s = random.randint(6, 10)  # Size parameter
    center = size // 2
    
    # Tam giác đều với đỉnh hướng lên
    pt1 = (center, center - s)           # Top vertex
    pt2 = (center - s, center + s)       # Bottom left
    pt3 = (center + s, center + s)       # Bottom right
    
    points = np.array([pt1, pt2, pt3], dtype=np.int32)
    color = random.randint(200, 255)  # White-ish color
    
    cv2.fillPoly(img, [points], color)
    
    # Light augmentation
    if random.random() > 0.7:
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(save_path, img)
    return img

def generate_dataset(num_samples_per_class=10000):
    """
    Generate dataset với:
    - train: 80% samples
    - val: 10% samples  
    - test: 10% samples
    """
    shapes = ['circle', 'square', 'triangle']
    generators = {
        'circle': generate_circle,
        'square': generate_square,
        'triangle': generate_triangle
    }
    
    # Calculate splits
    train_samples = int(num_samples_per_class * 0.8)
    val_samples = int(num_samples_per_class * 0.1)
    test_samples = num_samples_per_class - train_samples - val_samples
    
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    print(f"Generating {num_samples_per_class} images per class...")
    print(f"Split: train={train_samples}, val={val_samples}, test={test_samples}")
    
    for shape in shapes:
        print(f"\nGenerating {shape}...")
        generator_func = generators[shape]
        
        count = 0
        for split, num_samples in splits.items():
            # Create directory
            output_dir = os.path.join(ROOT_DIR, split, shape)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate images
            for i in range(num_samples):
                img_path = os.path.join(output_dir, f'{shape}_{count:05d}.png')
                generator_func(save_path=img_path)
                count += 1
            
            print(f"  - {split}: {num_samples} images")
    
    print("\n✅ Dataset generation complete!")
    print(f"Data saved in: {ROOT_DIR}")

if __name__ == '__main__':
    # Generate dataset
    generate_dataset(num_samples_per_class=1000)
    
    print("\nDataset structure:")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(ROOT_DIR, split)
        if os.path.exists(split_path):
            for shape in ['circle', 'square', 'triangle']:
                shape_path = os.path.join(split_path, shape)
                if os.path.exists(shape_path):
                    count = len([f for f in os.listdir(shape_path) if f.endswith('.png')])
                    print(f"  {split}/{shape}: {count} images")
