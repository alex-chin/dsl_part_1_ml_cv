# DATASET_PATH ="/kaggle/input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/"
# ATTRIBUTES_PATH = "/kaggle/input/lfw-attributes/lfw_attributes.txt"
DATASET_PATH = Path(images_path)
ATTRIBUTES_PATH = Path(attrs_path).joinpath("lfw_attributes.txt")
# DATASET_PATH = "C:\\Users\\k142\\.cache\\kagglehub\\datasets\\jessicali9530\\lfw-dataset\\versions\\4"
# ATTRIBUTES_PATH = "C:\\Users\\k142\\.cache\\kagglehub\\datasets\\averkij\\lfw-attributes\\versions\\1\\lfw_attributes.txt"

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class AutoencoderDataset(Dataset):
    def __init__(self, images, indices, transform=None):
        """
        Args:
            images: numpy array с изображениями
            indices: индексы для выборки
            transform: optional transform to be applied
        """
        self.images = images[indices]
        self.transform = transform

        # Нормализация данных (если нужно)
        self.images = self.images.astype(np.float32) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        # Для автоэнкодера и вход и выход - одно и то же изображение
        return image, image

def get_dataloaders(images, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    """
    Создает DataLoader'ы для train, validation и test

    Args:
        images: numpy array с изображениями (формат: N x H x W x C)
        batch_size: размер батча
        train_ratio: доля тренировочных данных
        val_ratio: доля валидационных данных

    Returns:
        train_loader, val_loader, test_loader
    """
    n_total = len(images)
    ix = np.random.choice(n_total, n_total, False)  # Перемешанные индексы

    train_size = int(train_ratio * n_total)
    val_size = int(val_ratio * n_total)

    tr, val, ts = np.split(ix, [train_size, train_size + val_size])

    # Создаем датасеты
    train_dataset = AutoencoderDataset(images, tr)
    val_dataset = AutoencoderDataset(images, val)
    test_dataset = AutoencoderDataset(images, ts)

    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


#%%
train_ratio = 0.8
val_ratio = 0.1
n_total = len(images)  # Общее количество данных
# test_ratio = 1 - train_ratio - val_ratio (автоматически)
ix = np.random.choice(n_total, n_total, False)  # Перемешанные индексы

train_size = int(train_ratio * n_total)
val_size = int(val_ratio * n_total)

tr, val, ts = np.split(ix, [train_size, train_size + val_size])