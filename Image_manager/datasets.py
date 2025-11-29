import os
import numpy as np
import pandas as pd

        
    
def load_data(data_path):
    channel_1 = []
    channel_2 =[]
    Y = []

    data_list = os.listdir(data_path)

    for folder in data_list:

        folder_path = os.path.join(data_path, folder)
        optir_path = os.path.join(folder_path, "optir")
        dc_path    = os.path.join(folder_path, "DC")
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


