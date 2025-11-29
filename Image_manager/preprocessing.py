import numpy as np
from skimage.transform import resize
import cv2 

def apply_mask(images, size=1500):
    images_plus_masks=[]
    for img in images:
        img = img.astype(np.float32)
        H, W = img.shape
        img = img.astype(np.float32) 
        mask = np.zeros((size, size),dtype=np.float32)
        offset_y = (size - H) // 2
        offset_x = (size - W) // 2
        mask[offset_y : offset_y + H, offset_x : offset_x + W] = img
        images_plus_masks.append(mask)
    return images_plus_masks

def merge (channel_1, channel_2, height_shape, width_shape):
    X=[]
    
    for i in range(len(channel_1)):
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
    

def invert_channel(images):
    # Convertimos todo a float pero SIN apilar
    imgs_np = [np.array(img, dtype=float) for img in images]
    # Máximo global entre todas las imágenes y todos los pixeles
    max_val = max(img.max() for img in imgs_np)
    #almacenar imagenes invertidas de acuerodo al maximo global
    inverted_images = []
    for img in imgs_np:
        inverted_images.append(max_val - img)

    return inverted_images


def apply_thresholding(images, threshold):
    #aplicar umbaral
    thresholded_images =[]
    #solo funciona para filtrar los valores menores al umbral, los demas valores los mantiene iguales
    for img in images:
        retval, channel_thresholded = cv2.threshold(img, thresh=threshold, maxval=2000, type=cv2.THRESH_TOZERO)
        thresholded_images.append(channel_thresholded)
        
    return thresholded_images


def global_normalize(images, low_percentile=1, high_percentile=99.99):
    """
    images: lista de arrays numpy (H, W)
    Este método calcula percentiles globales e intenta preservar la mayor cantidad de información.
    """

    # 1) Convertir lista → un solo array (N, H, W)
    stack = np.array(images)
    
    # 2) Obtener percentiles globales
    value_low_percentile = np.percentile(stack, low_percentile)
    value_high_percentile = np.percentile(stack, high_percentile)

    print(f"Usando rango global [{value_low_percentile:.3f}, {value_high_percentile:.3f}]")

    # 3) Clipping global
    stack_clipped = np.clip(stack, value_low_percentile, value_high_percentile)

    # 4) Normalización global a 0–1
    normalized = (stack_clipped - value_low_percentile) / (value_high_percentile - value_low_percentile)

    # 5) Volver a lista si lo necesitas
    normalized_list = [normalized[i] for i in range(len(images))]

    print(value_high_percentile)
    return normalized_list 


def clipping_log_normalize(images, low_percentile=2, high_percentile=99.99):
    
    #Aplica clipping + log-normalización a una lista de imágenes.
    normalized_list = []

    for img in images:
        img = np.array(img)

        # 1. Clipping por percentiles
        value_low_percentile = np.percentile(img, low_percentile)
        value_high_percentile = np.percentile(img, high_percentile)
        img_clip = np.clip(img, value_low_percentile, value_high_percentile)

        # 2. Transformación logarítmica
        img_log = np.log1p(img_clip)

        # 3. Normalización min–max
        img_norm = (img_log - img_log.min()) / (img_log.max() - img_log.min())

        normalized_list.append(img_norm)

    print(value_low_percentile, value_high_percentile)
    return normalized_list

def clipping_maxmin_normalize(images, low_percentile=0, high_percentile=100):
    normalized_list = []
    all_pixels = np.concatenate([img.ravel() for img in images])
    
    value_low_percentile = np.percentile(all_pixels, low_percentile)
    value_high_percentile = np.percentile(all_pixels, high_percentile)
    for img in images:
        img = np.array(img)
        img_clip = np.clip(img, value_low_percentile, value_high_percentile)

        # 3. Normalización min–max
        img_norm = (img_clip - img_clip.min()) / (img_clip.max() - img_clip.min())

        normalized_list.append(img_norm)

    print(value_high_percentile)
    return normalized_list


def normalize_sqrt(images):
    normalized = []
    # max global
    max_val = np.max([np.max(img) for img in images])
    
    for img in images:
        img_sqrt = np.sqrt(img)
        normalized.append(img_sqrt / np.sqrt(max_val))
    print(max_val)
    return normalized



def clipping_log_normalize(images, low_percentile=1, high_percentile=99.99):
    """
    Normaliza imágenes a [0,1] usando percentiles globales,
    excluyendo los ceros completamente del cálculo.
    """

    # 1) Extraer todos los píxeles diferentes de cero
    all_pixels = np.concatenate([img[img > 0].ravel() for img in images])

    if all_pixels.size == 0:
        raise ValueError("Todas las imágenes son cero.")

    # 2) Calcular percentiles globales sin incluir ceros
    value_low_percentile = np.percentile(all_pixels, low_percentile)
    value_high_percentile = np.percentile(all_pixels, high_percentile)

    print(f"Usando clipping global sin ceros [{value_low_percentile:.4f}, {value_high_percentile:.4f}]")

    logged_images = []
    logs_min = float("inf")
    logs_max = -float("inf")

    # 3) Primer pase (log + encontrar min/max ignorando ceros)
    for img in images:
        # hacer clipping pero dejando ceros intactos
        img_nonzero = img.copy()
        mask = img_nonzero > 0
        img_nonzero[mask] = np.clip(img_nonzero[mask], value_low_percentile, value_high_percentile)

        # aplicar log sólo donde img>0
        logged = np.zeros_like(img_nonzero, dtype=float)
        logged[mask] = np.log1p(img_nonzero[mask])
        logged_images.append(logged)

        if mask.any():
            logs_min = min(logs_min, logged[mask].min())
            logs_max = max(logs_max, logged[mask].max())

    if logs_max == logs_min:
        raise ValueError("Los valores no permiten normalización.")

    # 4) Normalización final (mantener ceros como ceros)
    normalized_list = []
    for logged in logged_images:
        norm = np.zeros_like(logged, dtype=float)
        mask = logged > 0
        norm[mask] = (logged[mask] - logs_min) / (logs_max - logs_min)
        normalized_list.append(norm)

    return normalized_list