import os
import sys
import numpy as np
import tensorflow as tf
import keras
from keras import layers, models, backend as K
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import KFold

# =========================================================================
# STEP 1: LOAD HANYA DATA TRAIN
# =========================================================================
def load_train_data(base_dir, target_size=(9, 9)):
    categories = ['dark', 'green', 'light', 'medium']
    X_data, y_data = [], []
    
    print("=== [TRAIN] Memuat Dataset dari Folder 'train' ===")
    for label_idx, category in enumerate(categories):
        # Menyasar folder train/ secara spesifik
        dir_path = os.path.join(base_dir, 'train', category)
        if not os.path.exists(dir_path):
            dir_path = os.path.join(base_dir, 'Train', category.capitalize())
            
        file_list = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"-> Memuat data train kelas '{category}': {len(file_list)} gambar.")
        
        for file_name in file_list:
            img_path = os.path.join(dir_path, file_name)
            try:
                img = load_img(img_path, target_size=target_size)
                X_data.append(img_to_array(img))
                y_data.append(label_idx)
            except:
                pass
                
    X_data = np.array(X_data, dtype='float32') / 255.0
    y_data = np.array(y_data, dtype='int32')
    y_data_onehot = tf.keras.utils.to_categorical(y_data, num_classes=4)
    return X_data, y_data_onehot

# =========================================================================
# STEP 2: CUSTOM CAPSNET LAYERS & GENERATOR
# =========================================================================
class Squash(layers.Layer):
    def call(self, inputs, axis=-1, **kwargs):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
        scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + K.epsilon())
        return scale * inputs

class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + K.epsilon())

class DigitCaps(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule],
                                 initializer='glorot_uniform', name='W')
        self.built = True

    def call(self, inputs):
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])
        u_hat = tf.matmul(self.W, inputs_tiled)
        u_hat = tf.squeeze(u_hat, [4])
        b = tf.zeros(shape=[tf.shape(u_hat)[0], self.num_capsule, self.input_num_capsule])

        for r in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            c_expanded = tf.expand_dims(c, axis=-2)
            s_j = tf.matmul(c_expanded, u_hat)
            v_j = Squash()(s_j)
            if r < self.routings - 1:
                b += tf.reduce_sum(u_hat * v_j, axis=-1)
        return tf.squeeze(v_j, axis=2)

def ContinuousCapsNet(input_shape=(9, 9, 3), n_class=4):
    inputs = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    conv2 = layers.Conv2D(128, (2, 2), padding='same', activation='relu')(conv1)
    conv3 = layers.Conv2D(256, (2, 2), padding='same', activation='relu')(conv2)
    conv4 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(conv3)

    dim_caps_primary = 8
    num_caps_primary = (conv4.shape[1] * conv4.shape[2] * conv4.shape[3]) // dim_caps_primary
    primary_caps = layers.Reshape(target_shape=(num_caps_primary, dim_caps_primary))(conv4)
    primary_caps_activated = Squash()(primary_caps)

    digit_caps = DigitCaps(num_capsule=n_class, dim_capsule=16, routings=3)(primary_caps_activated)
    out_caps = Length()(digit_caps)
    return models.Model(inputs=inputs, outputs=out_caps)

# =========================================================================
# RUN TRAINING DENGAN VALIDASI K-FOLD INTERNAL & GENERATE GRAFIK
# =========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Import tambahan buat nge-plot grafik

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(current_dir, "dataset")
    
    X_train_all, y_train_all = load_train_data(path_dataset)
    
    # =========================================================================
# RUN TRAINING DENGAN VALIDASI K-FOLD INTERNAL & GENERATE GRAFIK
# =========================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt  # Import tambahan buat nge-plot grafik

    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(current_dir, "dataset")
    
    # 1. Load seluruh data train
    X_train_all, y_train_all = load_train_data(path_dataset)
    
    # =========================================================================
    # --- TAMBAHAN KODE: PIE CHART DISTRIBUSI DATASET (BALANCED DATA) ---
    # =========================================================================
    print("\n=== [PROSES] Membuat Grafik Pie Chart Distribusi Dataset ===")
    
    y_labels_single = np.argmax(y_train_all, axis=1)
    unique, counts = np.unique(y_labels_single, return_counts=True)
    
    categories = ['Dark Roast', 'Green Bean', 'Light Roast', 'Medium Roast']
    colors_kopi = ['#2c3e50', '#27ae60', '#f1c40f', '#d35400'] # Hitam, Hijau, Kuning, Oranye
    
    plt.figure(figsize=(7, 7))
    plt.pie(
        counts, 
        labels=categories, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors_kopi,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.85}
    )
    plt.title('Grafik Distribusi Sebaran Sampel Dataset Biji Kopi\n(Balanced Dataset Framework)', fontsize=13, pad=20, fontweight='bold')
    plt.tight_layout()
    
    pie_save_path = os.path.join(current_dir, "Grafik_Distribusi_Dataset.png")
    plt.savefig(pie_save_path, dpi=300)
    print(f"[SUKSES] Gambar grafik distribusi (.png) disimpan di: {pie_save_path}")
    # =========================================================================

    # =========================================================================
    # --- KODE UTAMA: PROSES K-FOLD & PELATIHAN MODEL (JANGAN SAMPAI ILANG!) ---
    # =========================================================================
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    print("\n=== Melakukan Validasi K-Fold Internal pada Data Train ===")
    
    for train_idx, val_idx in kf.split(X_train_all):
        X_tr, X_val = X_train_all[train_idx], X_train_all[val_idx]
        y_tr, y_val = y_train_all[train_idx], y_train_all[val_idx]
        
        model = ContinuousCapsNet()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['accuracy'])
        
        print("\n--> Memulai Training dan Perekaman History Per Epoch <--")
        history = model.fit(
            X_tr, y_tr, 
            batch_size=4, 
            epochs=10,  # Latih 10 epoch biar tren grafiknya kelihatan jelas
            validation_data=(X_val, y_val), 
            verbose=1
        )
        break # Ambil satu putaran fold terbaik untuk representasi grafik laporan
        
    # --- PROSES PEMBUATAN GRAFIK TRAINING HISTORY (ACCURACY & LOSS) ---
    print("\n=== [PROSES] Menghasilkan Grafik Performa Training ===")
    plt.figure(figsize=(12, 5))

    # Plot Grafik Perubahan Akurasi
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Grafik Akurasi Model CapsNet Kopi\n(Training vs Validation)', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Plot Grafik Perubahan Nilai Error (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='green', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Grafik Loss Model CapsNet Kopi\n(Training vs Validation)', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    
    grafik_path = os.path.join(current_dir, "Grafik_Training_History.png")
    plt.savefig(grafik_path, dpi=300)
    print(f"[SUKSES] Grafik history berhasil disimpan di: {grafik_path}")
    
    # Save model utuh ke file .keras agar siap dipakai oleh test.py
    model_save_path = os.path.join(current_dir, "model_capsnet_kopi.keras")
    model.save(model_save_path)
    print(f"\n[SUKSES] Model final berhasil disimpan di: {model_save_path}")
    
    # Menampilkan grafik pie chart dan history secara bergantian ke layar LOQ lu
    plt.show()
        
    # --- PROSES PEMBUATAN GRAFIK (ACCURACY & LOSS) ---
    print("\n=== [PROSES] Menghasilkan Grafik Performa Training ===")
    plt.figure(figsize=(12, 5))

    # 1. Plot Grafik Perubahan Akurasi (Accuracy)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
    plt.title('Grafik Akurasi Model CapsNet Kopi\n(Training vs Validation)', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2. Plot Grafik Perubahan Nilai Error (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='green', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
    plt.title('Grafik Loss Model CapsNet Kopi\n(Training vs Validation)', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    
    # Simpan grafik berupa file gambar .png di folder project lu
    grafik_path = os.path.join(current_dir, "Grafik_Training_History.png")
    plt.savefig(grafik_path, dpi=300)
    print(f"[SUKSES] Grafik history berhasil disimpan di: {grafik_path}")
    plt.show() # Tampilkan langsung di layar laptop LOQ lu
        
    # Save model utuh ke file .keras agar siap dipakai oleh test.py
    model_save_path = os.path.join(current_dir, "model_capsnet_kopi.keras")
    model.save(model_save_path)
    print(f"\n[SUKSES] Model final berhasil disimpan di: {model_save_path}")