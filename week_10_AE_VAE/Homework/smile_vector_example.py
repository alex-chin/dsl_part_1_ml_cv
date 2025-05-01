import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from smile_vector import get_smile_vectors, add_smile, visualize_smile_transformation

def load_data(images_path, attrs_path):
    """
    Загрузка данных изображений и атрибутов
    
    Args:
        images_path: путь к директории с изображениями
        attrs_path: путь к файлу с атрибутами
    
    Returns:
        images: numpy array изображений
        attrs: DataFrame с атрибутами
    """
    # Загрузка атрибутов
    attrs = pd.read_csv(attrs_path, sep='\t')
    attrs['Smiling'] = attrs['Smiling'].astype(float)
    
    # Здесь должна быть загрузка изображений
    # В реальном коде нужно реализовать загрузку изображений из директории
    # Для примера предположим, что images уже загружены
    print("Атрибуты загружены, размер DataFrame:", attrs.shape)
    return None, attrs  # Вместо None должен быть массив изображений

def main():
    # Пути к данным (нужно заменить на реальные пути)
    images_path = Path("path/to/images")
    attrs_path = Path("path/to/attributes.txt")
    
    # Загрузка данных
    images, attrs = load_data(images_path, attrs_path)
    
    # Проверка наличия данных
    if images is None:
        print("Ошибка: изображения не загружены")
        return
    
    # Создание и загрузка модели
    # В реальном коде здесь нужно загрузить предварительно обученную модель
    # Для примера создадим заглушку
    class DummyModel:
        def __init__(self):
            self.encoder = None
            self.decoder = None
        def to(self, device):
            return self
        def eval(self):
            pass
    
    model = DummyModel()
    print("Модель создана")
    
    # Получение вектора улыбки
    try:
        smile_vector, smiling_indices, non_smiling_indices = get_smile_vectors(model, images, attrs)
        print(f"Количество улыбающихся людей: {len(smiling_indices)}")
        print(f"Количество не улыбающихся людей: {len(non_smiling_indices)}")
        print(f"Размерность вектора улыбки: {smile_vector.shape}")
    except Exception as e:
        print(f"Ошибка при получении вектора улыбки: {e}")
        return
    
    # Пример добавления улыбки к одному изображению
    try:
        # Выбираем индекс не улыбающегося человека
        test_idx = non_smiling_indices[0]
        
        # Создаем сетку для визуализации
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Исходное изображение
        axes[0].imshow(images[test_idx])
        axes[0].set_title('Оригинал')
        axes[0].axis('off')
        
        # Добавляем улыбку с разной интенсивностью
        for i, alpha in enumerate([0.5, 1.0]):
            result = add_smile(model, images[test_idx], smile_vector, alpha=alpha)
            axes[i+1].imshow(result)
            axes[i+1].set_title(f'Улыбка (α={alpha})')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Ошибка при добавлении улыбки к одному изображению: {e}")
    
    # Визуализация нескольких примеров
    try:
        print("\nВизуализация трансформации для 5 изображений:")
        visualize_smile_transformation(model, images, attrs, n_samples=5)
    except Exception as e:
        print(f"Ошибка при визуализации нескольких примеров: {e}")

if __name__ == "__main__":
    main() 