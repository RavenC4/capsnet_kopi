import tensorflow as tf
from tensorflow.keras import layers, backend as K

class Squash(layers.Layer):
    """
    Fungsi Aktivasi Squashing (Persamaan 3.15)
    Memastikan panjang vektor output berada di rentang 0-1 untuk merepresentasikan probabilitas,
    tanpa mengubah arah/orientasi spasial vektornya.
    """
    def call(self, inputs, axis=-1, **kwargs):
        # Hitung ||s_j||^2
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
        # Hitung fungsi squashing: (||s_j||^2 / (1 + ||s_j||^2)) * (s_j / ||s_j||)
        scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + K.epsilon())
        return scale * inputs

class Length(layers.Layer):
    """
    Menghitung panjang/norm L2 dari vektor CapsNet (||v_j||)
    Panjang akhir ini digunakan untuk menentukan tingkat kematangan kelas yang diprediksi.
    """
    def call(self, inputs, **kwargs):
        # Menghitung akar dari penjumlahan kuadrat elemen vektor + epsilon (biar ga pembagian nol)
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + K.epsilon())

class DigitCaps(layers.Layer):
    """
    Digit/Emotion Capsules Layer dengan Algoritma Dynamic Routing (Tabel 3.2)
    Bertanggung jawab atas proses Affine Transformation dan Perutean informasi spasial.
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule    # Jumlah kelas target (4 kelas: dark, green, light, medium)
        self.dim_capsule = dim_capsule    # Dimensi vektor output capsule (16 neuron)
        self.routings = routings          # Jumlah iterasi routing r = 3 (sesuai standar Sabour/slide)

    def build(self, input_shape):
        # input_shape: [Batch_Size, Jumlah_Capsule_Bawah (648), Dimensi_Capsule_Bawah (8)]
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # 1. Inisialisasi awal matriks bobot W_ij berukuran 16 x 8 (Persamaan 3.12 / Slide Hal 12)
        self.W = self.add_weight(
            shape=[self.num_capsule, self.input_num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )
        self.built = True

    def call(self, inputs):
        # == PROSES AFERIN TRANSFORMATION (u_hat = W_ij * u_i) ==
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 1), -1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1, 1])
        u_hat = tf.matmul(self.W, inputs_tiled)
        u_hat = tf.squeeze(u_hat, [4]) # Vektor prediksi u_hat didapat

        # == ALGORITMA DYNAMIC ROUTING (Tabel 3.1 / Slide Hal 4) ==
        # Baris 2: Inisialisasi awal nilai b_ij = 0 (Null) untuk semua capsule i dan j [cite: 304, 309]
        b = tf.zeros(shape=[tf.shape(u_hat)[0], self.num_capsule, self.input_num_capsule])

        # Baris 3: Loop sebanyak r kali iterasi routing [cite: 304, 310]
        for r in range(self.routings):
            # Baris 4: Menghitung Coupling Coefficient c_ij dengan Softmax [cite: 304, 312]
            c = tf.nn.softmax(b, axis=1)
            
            # --- PERBAIKAN DI SINI ---
            # Menyelaraskan dimensi c agar kompatibel dengan u_hat untuk proses perkalian matriks
            c_expanded = tf.expand_dims(c, axis=-2) # Dimensi berubah menjadi [?, 4, 1, 648]
            
            # Baris 5: Proses Weighted Sum (s_j = jumlah dari c_ij * u_hat) 
            # Melakukan perkalian matriks secara broadcasted batch dot product
            s_j = tf.matmul(c_expanded, u_hat) # Hasil dimensi s_j: [?, 4, 1, 16]
            # -------------------------
            
            # Baris 6: Aktivasi vektor s_j dengan fungsi Squashing untuk menghasilkan v_j [cite: 304, 328, 329]
            v_j = Squash()(s_j)

            # Baris 7: Perbarui nilai kedekatan b_ij sebelum iterasi terakhir [cite: 304, 333]
            if r < self.routings - 1:
                # Dot product antara u_hat dan v_j untuk meng-update bobot kecocokan rute (b) [cite: 304, 335]
                u_v_dot = tf.reduce_sum(u_hat * v_j, axis=-1) # Hasil dimensi: [?, 4, 648]
                b += u_v_dot

        # Baris 8: Kembalikan nilai akhir v_j [cite: 304]
        return tf.squeeze(v_j, axis=2)