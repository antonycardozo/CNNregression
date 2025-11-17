import os
import numpy as np
from tqdm import tqdm 
from skimage.transform import resize
import pandas as pd


def load_data(data_path, height_shape=128, width_shape=128):
    X = []

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
            img_optir = pd.read_csv(optir_excel, header=None).to_numpy()
            # Redimensionar a height × width
            img_optir = resize(img_optir,(height_shape, width_shape), preserve_range=True, anti_aliasing=True)
            # Añadir canal 
            img_optir = img_optir.reshape(height_shape, width_shape, 1)
            
            # ==== DC =====
            dc_excel = os.path.join(dc_path, fname)
            img_dc = pd.read_csv(dc_excel, header=None).to_numpy()
            img_dc = resize(img_dc,(height_shape, width_shape), preserve_range=True, anti_aliasing=True)
            img_dc = img_dc.reshape(height_shape, width_shape, 1)
            
            merged = np.concatenate([img_dc, img_optir], axis=-1)
            X.append(merged)
            # === LABEL desde el nombre ===
            label = int(fname.split("_")[0])
            Y.append(label)
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=int)

    return X, Y

