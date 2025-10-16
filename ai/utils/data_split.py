import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = r"D:\PTIT\XLA\data\train"
DEST_DIR   = r"D:\PTIT\XLA\data_split"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for subset in ["train", "val", "test"]:
    for cls in os.listdir(SOURCE_DIR):
        os.makedirs(os.path.join(DEST_DIR, subset, cls), exist_ok=True)

# Chia dữ liệu
for cls in tqdm(os.listdir(SOURCE_DIR), desc="Splitting classes"):
    src_folder = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src_folder)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    n_test  = n_total - n_train - n_val

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train + n_val]
    test_imgs  = images[n_train + n_val:]

    for name, subset in [(train_imgs, "train"), (val_imgs, "val"), (test_imgs, "test")]:
        for img_name in name:
            src = os.path.join(src_folder, img_name)
            dst = os.path.join(DEST_DIR, subset, cls, img_name)
            shutil.copy2(src, dst)
