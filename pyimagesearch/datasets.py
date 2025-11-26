import os
import numpy as np
from tqdm import tqdm 
from skimage.transform import resize
import pandas as pd
import cv2 

def apply_mask(imagen, size=1500):
    masks=[]
    for img in imagen:
        img = img.astype(np.float32)
        H, W = img.shape
        img = img.astype(np.float32) 
        mask = np.zeros((size, size),dtype=np.float32)
        offset_y = (size - H) // 2
        offset_x = (size - W) // 2
        mask[offset_y : offset_y + H, offset_x : offset_x + W] = img
        masks.append(mask)
    return masks

def preprocessing (channel_1, channel_2, height_shape, width_shape):
    X=[]
    
    for i in range(len(channel_1)):
        # aplicar máscara

        # Redimensionar a height × width
        img_optir = resize(channel_1[i],(height_shape, width_shape), preserve_range=True, anti_aliasing=True)
        img_dc = resize(channel_2[i],(height_shape, width_shape), preserve_range=True, anti_aliasing=True)
        # Añadir canal 
        img_optir = img_optir.reshape(height_shape, width_shape, 1)
        img_dc = img_dc.reshape(height_shape, width_shape, 1)
        #unir imagenes
        merged = np.concatenate([img_dc, img_optir], axis=-1)
        X.append(merged)
    X = np.array(X, dtype=np.float32)
    return X            
    
import numpy as np

def filter_images(imagenes, greater_than=True, threshold=4.1):
    filtered_images = []

    for img in imagenes:

        # Caso 1: imagen 2D -> (H, W)
        if img.ndim == 2:
            if greater_than:
                mask = np.all(img > threshold, axis=0)
            else:
                mask = np.all(img < threshold, axis=0)

            mask = ~mask  # mantenemos las columnas que NO cumplen la condición
            img_filtrada = img[:, mask]

        # Caso 2: imagen 3D -> (H, W, C)
        elif img.ndim == 3:
            if greater_than:
                mask = np.all(img > threshold, axis=0)
            else:
                mask = np.all(img < threshold, axis=0)

            mask = ~mask
            img_filtrada = img[:, mask, :]

        else:
            raise ValueError(f"Formato de imagen no soportado: {img.shape}")

        filtered_images.append(img_filtrada)

    return filtered_images

def invert_DC(images):
    # Convertimos todo a float pero SIN apilar
    imgs_np = [np.array(img, dtype=float) for img in images]

    # Máximo global entre todas las imágenes y todos los pixeles
    max_val = max(img.max() for img in imgs_np)

    inverted = []
    for img in imgs_np:
        inverted.append(max_val - img)

    return inverted
    # image_i = ~ image_src 
    return invertes_images

def apply_thresholding(image_file, threshold):
    thresh_images =[]
    for img in image_file:
        retval, channel_wit_threshold = cv2.threshold(img, thresh=threshold, maxval=2000, type=cv2.THRESH_TOZERO)
        thresh_images.append(channel_wit_threshold)
        
    return thresh_images
        
    
def load_data(data_path, height_shape=128, width_shape=128):
    channel_1 = []
    channel_2 =[]
    Y = []

    data_list = os.listdir(data_path)

    for folder in tqdm(data_list):

        folder_path = os.path.join(data_path, folder)
        optir_path = os.path.join(folder_path, "optir")
        print(optir_path)
        dc_path    = os.path.join(folder_path, "DC")
        print(dc_path)
        # archivos compartidos
        file_list = os.listdir(optir_path)
        for fname in file_list:
            # ==== OPTIR =====
            optir_excel = os.path.join(optir_path, fname)
            img_optir = pd.read_csv(optir_excel, header=None).to_numpy(dtype=np.float32)
            channel_1.append(img_optir)
            # ==== DC =====
            dc_excel = os.path.join(dc_path, fname)
            img_dc = pd.read_csv(dc_excel, header=None).to_numpy(dtype=np.float32)
            channel_2.append(img_dc)
            # === LABEL desde el nombre ===
            label = float(fname.split("_")[0])
            Y.append(label)
    
    Y = np.array(Y, dtype=float)

    return channel_1,channel_2, Y



def global_normalize(images, low_percentile=1, high_percentile=99.99):
    """
    images: lista de arrays numpy (H, W)
    Este método calcula percentiles globales e intenta preservar la mayor cantidad de información.
    """

    # 1) Convertir lista → un solo array (N, H, W)
    stack = np.array(images)
    
    # 2) Obtener percentiles globales
    p_low = np.percentile(stack, low_percentile)
    p_high = np.percentile(stack, high_percentile)

    print(f"Usando rango global [{p_low:.3f}, {p_high:.3f}]")

    # 3) Clipping global
    stack_clipped = np.clip(stack, p_low, p_high)

    # 4) Normalización global a 0–1
    normalized = (stack_clipped - p_low) / (p_high - p_low)

    # 5) Volver a lista si lo necesitas
    normalized_list = [normalized[i] for i in range(len(images))]

    print(p_high)
    return normalized_list 


def clipping_log_norma(images, low=2, high=99.99):
    """
    Aplica clipping + log-normalización a una lista de imágenes.
    
    Parámetros:
        images: lista de arrays numpy (2D o 3D)
        low, high: percentiles para clipping
    
    Retorna:
        lista de imágenes normalizadas entre 0 y 1
    """

    normalized_list = []

    for img in images:
        img = np.array(img)

        # 1. Clipping por percentiles
        p_low = np.percentile(img, low)
        p_high = np.percentile(img, high)
        img_clip = np.clip(img, p_low, p_high)

        # 2. Transformación logarítmica
        img_log = np.log1p(img_clip)

        # 3. Normalización min–max
        img_norm = (img_log - img_log.min()) / (img_log.max() - img_log.min())

        normalized_list.append(img_norm)

    print(p_high)
    return normalized_list

def clipping_maxmin_norma(images, low=3, high=99.99):
    normalized_list = []
    all_pixels = np.concatenate([img.ravel() for img in images])
    
    p_low = np.percentile(all_pixels, low)
    p_high = np.percentile(all_pixels, high)
    for img in images:
        img = np.array(img)
        img_clip = np.clip(img, p_low, p_high)

        # 3. Normalización min–max
        img_norm = (img_clip - img_clip.min()) / (img_clip.max() - img_clip.min())

        normalized_list.append(img_norm)

    print(p_high)
    return normalized_list

def log_normalize(images):
    """
    Normaliza una lista de imágenes usando escala logarítmica
    y luego min-max global.
    Retorna:
      - lista de imágenes normalizadas en rango 0-1
      - min_val y max_val globales
    """
    # 1) Aplanar todo para calcular min y max globales
    all_pixels = np.concatenate([img.ravel() for img in images])
    
    # 2) Aplicar log1p globalmente
    log_all = np.log1p(all_pixels)

    min_val = log_all.min()
    max_val = log_all.max()

    # 3) Normalizar cada imagen usando ese min/max
    normalized_images = []
    for img in images:
        log_img = np.log1p(img)
        norm_img = (log_img - min_val) / (max_val - min_val)
        normalized_images.append(norm_img)

    return normalized_images


def normalize_sqrt(images):
    normalized = []
    # max global
    max_val = np.max([np.max(img) for img in images])
    
    for img in images:
        img_sqrt = np.sqrt(img)
        normalized.append(img_sqrt / np.sqrt(max_val))
    print(max_val)
    return normalized

import numpy as np

def global_log_normalize(images, low_percentile=1, high_percentile=99.99):
    """
    Normaliza imágenes de distinto tamaño a [0,1] usando percentiles globales.
    """

    # --- 1. Obtener todos los píxeles para los percentiles globales ---
    all_pixels = np.concatenate([img.ravel() for img in images])
    p_low = np.percentile(all_pixels, low_percentile)
    p_high = np.percentile(all_pixels, high_percentile)

    print(f"Usando clipping global [{p_low:.4f}, {p_high:.4f}]")

    # --- 2. Primer pase: calcular min y max globales después del log ---
    logs_min = float("inf")
    logs_max = -float("inf")

    logged_images = []  # guardamos temporalmente para evitar recomputación

    for img in images:
        clipped = np.clip(img, p_low, p_high)
        logged = np.log1p(clipped)
        logged_images.append(logged)

        # actualizar extremos globales
        logs_min = min(logs_min, logged.min())
        logs_max = max(logs_max, logged.max())

    # Evitar división entre cero
    if logs_max == logs_min:
        raise ValueError("Los valores después del log son constantes. No es posible normalizar.")

    # --- 3. Normalización final (global min-max) ---
    normalized_list = [
        (logged - logs_min) / (logs_max - logs_min)
        for logged in logged_images
    ]

    return normalized_list
