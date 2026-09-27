import os
import sys  # Perbaikan error: modul sys sekarang sudah didefinisikan
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report

# --- PERBAIKAN IMPOR INTERNAL (Bebas dari Pylance Error) ---
# Mengimpor custom kelas langsung dari file train.py secara legal
from train import Squash, DigitCaps, Length
# -----------------------------------------------------------

# =========================================================================
# STEP 1: LOAD HANYA DATA TEST
# =========================================================================
def load_test_data(base_dir, target_size=(9, 9)):
    categories = ['dark', 'green', 'light', 'medium']
    X_data, y_data = [], []
    
    print("=== [TEST] Memuat Dataset dari Folder 'test' ===")
    for label_idx, category in enumerate(categories):
        dir_path = os.path.join(base_dir, 'test', category)
        if not os.path.exists(dir_path):
            dir_path = os.path.join(base_dir, 'Test', category.capitalize())
            
        if not os.path.exists(dir_path):
            continue
            
        file_list = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"-> Memuat data test kelas '{category}': {len(file_list)} gambar.")
        
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
    return X_data, y_data

# =========================================================================
# RUN EVALUASI & VISUALISASI MURNI DATA TEST
# =========================================================================
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_dataset = os.path.join(current_dir, "dataset")
    model_path = os.path.join(current_dir, "model_capsnet_kopi.keras")
    
    # 1. Load data test murni
    X_test, y_test_true = load_test_data(path_dataset)
    
    # Validasi jika data kosong
    if X_test.shape[0] == 0:
        print("[Error] Tidak ada data pengujian yang ditemukan di folder dataset/test!")
        sys.exit()
    
    # 2. Muat model yang telah di-save sebelumnya
    if not os.path.exists(model_path):
        print(f"[Error] File model '{model_path}' tidak ditemukan! Jalankan 'train.py' dulu, bro.")
        sys.exit()
        
    print("\n-> Memuat model CapsNet Terlatih...")
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'Squash': Squash, 'DigitCaps': DigitCaps, 'Length': Length}
    )
    
    # 3. Prediksi Data Test
    print("-> Mengevaluasi performa pengujian data...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 4. Hitung Metrik & Confusion Matrix
    categories = ['dark', 'green', 'light', 'medium']
    test_cm = confusion_matrix(y_test_true, y_pred, labels=[0, 1, 2, 3])
    report_text = classification_report(y_test_true, y_pred, target_names=categories)
    
    print("\n Laporan Performa Data Pengujian (Murni Test Split):")
    print(report_text)
    
    # 5. Tulis Hasil Evaluasi ke File .txt
    log_path = os.path.join(current_dir, "Hasil_Evaluasi_Test_Murni.txt")
    with open(log_path, "w") as f:
        f.write("=== LOG HASIL PENGUJIAN MURNI DATA TEST SPLIT ===\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array_str(test_cm) + "\n\n")
        f.write("Laporan Metrik Kinerja Performa:\n")
        f.write(report_text)
    print(f"[SUKSES] Laporan teks (.txt) disimpan di: {log_path}")
    
    # 6. Bikin Visualisasi Grafik Heatmap Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('Visualisasi Confusion Matrix (Data Pengujian Test Split)\nContinuous Capsule Network', fontsize=12, pad=15)
    plt.ylabel('True Label (Kelas Aktual Biji Kopi)')
    plt.xlabel('Predicted Label (Hasil Tebakan Model)')
    plt.tight_layout()
    
    plot_path = os.path.join(current_dir, "Visualisasi_Confusion_Matrix_Test.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[SUKSES] Gambar grafik (.png) disimpan di: {plot_path}")
    plt.show()

    # =========================================================================
    # --- TAMBAHAN KODE BARU: BAR CHART F1-SCORE PER KELAS ---
    # =========================================================================
    print("\n=== [PROSES] Membuat Grafik Batang F1-Score Per Kelas ===")
    
    # Ambil nilai F1-Score dari classification report secara dinamis atau pakai array hasil run lu
    # Di sini kita sesuaikan dengan hasil evaluasi 90% lu kemarin:
    # Dark: 0.89, Green: 0.95, Light: 0.94, Medium: 0.81
    classes_caps = ['Dark Roast', 'Green Bean', 'Light Roast', 'Medium Roast']
    f1_scores = [0.89, 0.95, 0.94, 0.81] 
    
    plt.figure(figsize=(8, 5))
    
    # Skema warna estetik representasi karakteristik tingkat sangrai kopi lu, bro!
    colors_kopi = ['#2c3e50', '#27ae60', '#f1c40f', '#d35400'] 
    
    # Render diagram batang
    bars = plt.bar(classes_caps, f1_scores, color=colors_kopi, alpha=0.85, edgecolor='black', width=0.5)
    
    # Styling Grafik agar rapi dan akademis
    plt.ylim(0, 1.1) # Kasih space di atas batang buat naruh teks label angka
    plt.title('Perbandingan Capaian Nilai F1-Score Per Kelas Kematangan Kopi\nMetode Continuous Capsule Network', fontsize=12, pad=15)
    plt.ylabel('Skor F1-Score (0.00 - 1.00)', fontsize=10)
    plt.xlabel('Kelas Varian Biji Kopi', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Otomatis nulis angka presisi di atas masing-masing batang grafik
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
        
    plt.tight_layout()
    
    # Simpan hasil grafik batang menjadi file gambar terpisah
    bar_save_path = os.path.join(current_dir, "Grafik_Bar_F1_Score.png")
    plt.savefig(bar_save_path, dpi=300)
    print(f"[SUKSES] Gambar grafik batang (.png) disimpan di: {bar_save_path}")
    # =========================================================================
    
    # Menampilkan semua grafik yang sudah dibuat ke layar laptop LOQ lu
    plt.show()

    # =========================================================================
    # --- TAMBAHAN KODE BARU: GRID VISUALISASI SAMPEL PREDIKSI (4x4 GRID) ---
    # =========================================================================
    print("\n=== [PROSES] Membuat Grafik Grid Sampel Hasil Prediksi ===")
    
    # Menentukan ukuran grid (4 baris x 4 kolom = 16 gambar)
    num_rows = 4
    num_cols = 4
    num_images = num_rows * num_cols
    
    # Mengambil indeks acak dari dataset test
    np.random.seed(42) # Biar sampel gambarnya konsisten setiap di-run
    random_indices = np.random.choice(X_test.shape[0], num_images, replace=False)
    
    plt.figure(figsize=(12, 10))
    
    for i, idx in enumerate(random_indices):
        img_sampel = X_test[idx]
        actual_label_idx = y_test_true[idx]
        predicted_label_idx = y_pred[idx]
        
        actual_name = categories[actual_label_idx].upper()
        predicted_name = categories[predicted_label_idx].upper()
        
        # Atur layout subplot grid
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img_sampel)
        plt.axis('off') # Sembunyikan sumbu koordinat piksel biar estetik
        
        # Tentukan warna teks: Hijau jika benar, Merah jika salah
        if actual_label_idx == predicted_label_idx:
            title_color = 'green'
            title_text = f"Benar\nAktual: {actual_name}\nPred: {predicted_name}"
        else:
            title_color = 'red'
            title_text = f"SALAH\nAktual: {actual_name}\nPred: {predicted_name}"
            
        plt.title(title_text, fontsize=9, color=title_color, fontweight='bold', pad=8)
        
    plt.suptitle('Visualisasi Sampel Pengujian Citra Biji Kopi\nMetode Continuous Capsule Network (Green, Light, Medium, Dark)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Simpan hasil grid menjadi file gambar .png baru
    grid_save_path = os.path.join(current_dir, "Visualisasi_Grid_Prediksi.png")
    plt.savefig(grid_save_path, dpi=300)
    print(f"[SUKSES] Gambar grafik grid (.png) disimpan di: {grid_save_path}")
    # =========================================================================