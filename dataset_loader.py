import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_kopi_dataset(base_dir, target_size=(9, 9)):
    """
    Fungsi modular untuk memuat dataset citra biji kopi.
    Otomatis menyisir folder 'train' dan 'test' untuk menggabungkan data
    ke dalam satu kesatuan array demi kebutuhan K-Fold Cross Validation.
    """
    # Sesuaikan dengan nama folder asli lu (sensitif huruf besar/kecil)
    categories = ['Dark', 'Green', 'Light', 'Medium']
    # Sub-folder split di dalam direktori dataset lu
    splits = ['train', 'test']
    
    X_data = []
    y_data = []
    
    print("=== [STEP 1] Memulai Proses Pengolahan Dataset Kopi (Train & Test Split) ===")
    
    for label_idx, category in enumerate(categories):
        total_gambar_kelas = 0
        
        # Menyisir folder train dan test sekaligus untuk kelas yang sama
        for split in splits:
            dir_path = os.path.join(base_dir, split, category)
            
            if not os.path.exists(dir_path):
                # Antisipasi kalau foldernya pakai huruf kapital awal (e.g., 'Dark', 'Train')
                dir_path_cap = os.path.join(base_dir, split.capitalize(), category.capitalize())
                if os.path.exists(dir_path_cap):
                    dir_path = dir_path_cap
                else:
                    continue
            
            # Filter hanya file gambar (.png, .jpg, .jpeg)
            file_list = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_gambar_kelas += len(file_list)
            
            for file_name in file_list:
                img_path = os.path.join(dir_path, file_name)
                try:
                    # Load gambar & paksa resize ke 9x9 piksel sesuai slide materi
                    img = load_img(img_path, target_size=target_size)
                    img_array = img_to_array(img)
                    
                    X_data.append(img_array)
                    y_data.append(label_idx)
                except Exception as e:
                    print(f"   [Error] Gagal membaca gambar {file_name}: {str(e)}")
                    
        print(f"-> Menggabungkan data kelas '{category}': Total {total_gambar_kelas} gambar ditemukan.")
                
    # Konversi hasil list menjadi Numpy Array agar siap diolah TensorFlow
    X_data = np.array(X_data, dtype='float32')
    y_data = np.array(y_data, dtype='int32')
    
    # Normalisasi piksel citra (Mengubah rentang nilai 0-255 menjadi 0.0-1.0)
    X_data = X_data / 255.0
    
    # Konversi label skalar menjadi format One-Hot Encoding (4 Kelas)
    y_data_onehot = tf.keras.utils.to_categorical(y_data, num_classes=4)
    
    print(f"\n=== [SUKSES] Seluruh data train & test berhasil digabungkan! ===")
    print(f"Total Sampel Data : {X_data.shape[0]}")
    print(f"Dimensi Array Data: {X_data.shape} -> (Jumlah_Sampel, Tinggi, Lebar, Channel)")
    print(f"Dimensi Label     : {y_data_onehot.shape} -> (Jumlah_Sampel, Jumlah_Kelas)\n")
    
    return X_data, y_data_onehot

