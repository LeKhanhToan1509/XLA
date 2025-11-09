# ai/main.py
from ai.training.train import train_model
from ai.utils.generate_shape import generate_dataset
from ai.utils.data_split import split_mnist

if __name__ == "__main__":
    model = train_model()
    print("✅ Hoàn tất! Model lưu tại best_resnet18.pth")