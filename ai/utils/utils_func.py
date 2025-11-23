import struct
import numpy as np
import math
import random
import os
from PIL import Image

# read the images and labels from MNIST binary format
def readDataset(dataset):
    (image, label) = dataset
    with open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (img, lbl)

# read images from folder structure (e.g., data/train/0/, data/train/1/, ...)
def readDatasetFromFolder(folder_path):
    """
    Đọc ảnh từ cấu trúc thư mục:
    folder_path/
        0/
            image1.png
            image2.png
        1/
            image1.png
        ...
        
    Returns:
        images: numpy array shape (n_samples, height, width)
        labels: numpy array shape (n_samples,)
    """
    images = []
    labels = []
    
    # Lấy danh sách các class folders (0, 1, 2, ..., 9, circle, square, triangle)
    class_folders = sorted([f for f in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, f))])
    
    # Tạo mapping từ tên folder sang label number
    class_to_label = {}
    for idx, class_name in enumerate(class_folders):
        if class_name.isdigit():
            class_to_label[class_name] = int(class_name)
        else:
            # Với các class không phải số (circle, square, triangle)
            # Gán label từ 10 trở đi
            class_to_label[class_name] = 10 + idx - len([c for c in class_folders if c.isdigit()])
    
    print(f"Class mapping: {class_to_label}")
    
    # Đọc ảnh từ từng class folder
    for class_name in class_folders:
        class_path = os.path.join(folder_path, class_name)
        label = class_to_label[class_name]
        
        # Đọc tất cả ảnh trong folder
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Đọc ảnh và convert sang grayscale
                img = Image.open(img_path).convert('L')
                # Resize về 28x28 nếu cần
                if img.size != (28, 28):
                    img = img.resize((28, 28), Image.LANCZOS)
                # Convert sang numpy array
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
    
    images = np.array(images)
    labels = np.array(labels, dtype=np.int8)
    
    print(f"Loaded {len(images)} images from {folder_path}")
    print(f"Image shape: {images[0].shape if len(images) > 0 else 'N/A'}")
    
    return images, labels

# padding for the matrix of images
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))    
    return X_pad

# normalization of the input images
def normalize(image, mode='lenet5'):
    image -= image.min()
    image = image / image.max()
    # range = [0,1]
    if mode == '0p1':
        return image
    # range = [-1,1]
    elif mode == 'n1p1':
        image = image * 2 - 1
    # range = [-0.1,1.175]   
    elif mode == 'lenet5':
        image = image * 1.275 - 0.1
    return image

# initialization of the weights & bias
def initialize(kernel_shape, mode='Fan-in'):
    b_shape = (1,1,1,kernel_shape[-1]) if len(kernel_shape)==4 else (kernel_shape[-1],)
    if mode == 'Gaussian_dist':
        mu, sigma = 0, 0.1
        weight = np.random.normal(mu, sigma,  kernel_shape) 
        bias   = np.ones(b_shape)*0.01
        
    elif mode == 'Fan-in': #original init. in the paper
        Fi = np.prod(kernel_shape)/kernel_shape[-1]
        weight = np.random.uniform(-2.4/Fi, 2.4/Fi, kernel_shape)    
        bias   = np.ones(b_shape)*0.01     
    return weight, bias

# update for the weights
def update(weight, bias, dW, db, vw, vb, lr, momentum=0, weight_decay=0):
    vw_u = momentum*vw - weight_decay*lr*weight - lr*dW
    vb_u = momentum*vb - weight_decay*lr*bias   - lr*db
    weight_u = weight + vw_u
    bias_u   = bias   + vb_u
    return weight_u, bias_u, vw_u, vb_u 

# return random-shuffled mini-batches
def random_mini_batches(image, label, mini_batch_size = 256, one_batch=False):
    m = image.shape[0]                  # number of training examples
    mini_batches = []
    
    # Shuffle (image, label)
    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation,:,:,:]
    shuffled_label = label[permutation]
    
    # extract only one batch
    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)

    # Partition (shuffled_image, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    return mini_batches
