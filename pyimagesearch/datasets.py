import os
import numpy as np
from tqdm import tqdm 
from skimage.transform import resize
import pandas as pd

def apply_mask_1500(img):
    TARGET = 1500
    H, W = img.shape
    img = img.astype(np.float32) 
    mask = np.zeros((TARGET, TARGET),dtype=np.float32)

    offset_y = (TARGET - H) // 2
    offset_x = (TARGET - W) // 2

    mask[offset_y : offset_y + H, offset_x : offset_x + W] = img
    return mask

def preprocessing (channel_1, channel_2, height_shape, width_shape):
    X=[]
    
    for i in range(len(channel_1)):
        # aplicar máscara
        img_optir = apply_mask_1500(channel_1[i]).astype(np.float32)
        img_dc    = apply_mask_1500(channel_2[i]).astype(np.float32)
        # Redimensionar a height × width
        img_optir = resize(img_optir,(height_shape, width_shape), preserve_range=True, anti_aliasing=True)
        img_dc = resize(img_dc,(height_shape, width_shape), preserve_range=True, anti_aliasing=True)
        # Añadir canal 
        img_optir = img_optir.reshape(height_shape, width_shape, 1)
        img_dc = img_dc.reshape(height_shape, width_shape, 1)
        #unir imagenes
        merged = np.concatenate([img_dc, img_optir], axis=-1)
        X.append(merged)
    X = np.array(X, dtype=np.float32)
    return X            
    

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
            label = int(fname.split("_")[0])
            Y.append(label)
    
    Y = np.array(Y, dtype=int)

    return channel_1,channel_2, Y
