import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def get_smile_vectors(model, images, attrs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Вычисляет вектор улыбки на основе латентных представлений изображений
    
    Args:
        model: обученная модель автоэнкодера
        images: numpy array изображений формата (N, H, W, C)
        attrs: pandas DataFrame с атрибутами
        device: устройство для вычислений
    
    Returns:
        smile_vector: вектор улыбки (разность между средними векторами)
        smiling_indices: индексы улыбающихся людей
        non_smiling_indices: индексы не улыбающихся людей
    """
    # 1. Выбираем индексы улыбающихся и не улыбающихся людей
    smiling_indices = attrs.loc[attrs.Smiling > 2, 'Smiling'][:15].index.to_list()
    non_smiling_indices = attrs.loc[attrs.Smiling < -2, 'Smiling'][:15].index.to_list()
    
    # Преобразуем изображения в тензоры и меняем порядок размерностей NHWC -> NCHW
    smiling_images = torch.FloatTensor(images[smiling_indices]).permute(0, 3, 1, 2) / 255.0
    non_smiling_images = torch.FloatTensor(images[non_smiling_indices]).permute(0, 3, 1, 2) / 255.0
    
    # Перемещаем на нужное устройство
    model = model.to(device)
    smiling_images = smiling_images.to(device)
    non_smiling_images = non_smiling_images.to(device)
    
    # 2. Получаем латентные векторы
    model.eval()
    with torch.no_grad():
        # Для улыбающихся
        _, smiling_latent = model(smiling_images)
        # Для не улыбающихся
        _, non_smiling_latent = model(non_smiling_images)
    
    # 3. Вычисляем средние векторы и их разность
    mean_smiling = torch.mean(smiling_latent, dim=0)
    mean_non_smiling = torch.mean(non_smiling_latent, dim=0)
    smile_vector = mean_smiling - mean_non_smiling
    
    return smile_vector, smiling_indices, non_smiling_indices

def add_smile(model, image, smile_vector, alpha=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Добавляет улыбку к изображению
    
    Args:
        model: обученная модель автоэнкодера
        image: исходное изображение (numpy array формата H x W x C)
        smile_vector: вектор улыбки
        alpha: коэффициент интенсивности улыбки
        device: устройство для вычислений
    
    Returns:
        numpy array: изображение с добавленной улыбкой
    """
    # Преобразуем изображение в тензор и меняем порядок размерностей HWC -> CHW
    image_tensor = torch.FloatTensor(image).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    image_tensor = image_tensor.to(device)
    model = model.to(device)
    smile_vector = smile_vector.to(device)
    
    # Получаем латентный вектор изображения
    model.eval()
    with torch.no_grad():
        _, latent = model(image_tensor)
        # Добавляем вектор улыбки
        new_latent = latent + alpha * smile_vector
        # Декодируем обратно в изображение
        generated_image, _ = model.decoder(new_latent)
    
    # Преобразуем обратно в numpy array и меняем порядок размерностей обратно CHW -> HWC
    result = generated_image.cpu().squeeze(0).permute(1, 2, 0).numpy()
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    return result

def visualize_smile_transformation(model, images, attrs, n_samples=5):
    """
    Визуализирует процесс добавления улыбки к изображениям
    
    Args:
        model: обученная модель автоэнкодера
        images: numpy array изображений
        attrs: pandas DataFrame с атрибутами
        n_samples: количество примеров для визуализации
    """
    import matplotlib.pyplot as plt
    
    # Получаем вектор улыбки
    smile_vector, _, _ = get_smile_vectors(model, images, attrs)
    
    # Выбираем несколько не улыбающихся людей
    non_smiling_indices = attrs.loc[attrs.Smiling < -2, 'Smiling'][:n_samples].index.to_list()
    
    # Создаем сетку изображений
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    
    for i, idx in enumerate(non_smiling_indices):
        # Исходное изображение
        axes[i, 0].imshow(images[idx])
        axes[i, 0].set_title('Оригинал')
        axes[i, 0].axis('off')
        
        # Изображение с небольшой улыбкой
        with_smile_05 = add_smile(model, images[idx], smile_vector, alpha=0.5)
        axes[i, 1].imshow(with_smile_05)
        axes[i, 1].set_title('Улыбка (α=0.5)')
        axes[i, 1].axis('off')
        
        # Изображение с сильной улыбкой
        with_smile_10 = add_smile(model, images[idx], smile_vector, alpha=1.0)
        axes[i, 2].imshow(with_smile_10)
        axes[i, 2].set_title('Улыбка (α=1.0)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show() 