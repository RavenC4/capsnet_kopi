import tensorflow as tf
from tensorflow.keras import layers, backend as K

class Squash(layers.Layer):
    """
    Fungsi Aktivasi Squashing (Persamaan 3.15)
    Mengompresi panjang vektor ke rentang 0-1 tanpa mengubah arah/orientasi fitur spasial.
    """
    def call(self, inputs, axis=-1, **kwargs):
        squared_norm = tf.reduce_sum(tf.square(inputs), axis=axis, keepdims=True)
        scale = squared_norm / (1.0 + squared_norm) / tf.sqrt(squared_norm + K.epsilon())
        return scale * inputs

class Length(layers.Layer):
    """
    Menghitung panjang/norm L2 dari vektor output (||v_j||) untuk nilai probabilitas kelas.
    """
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1) + K.epsilon())

class DigitCaps(layers.Layer):
    """
    DigitCaps Layer dengan Mekanisme Dynamic Routing (r=3) yang Kompatibel dengan Keras 3+.
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(DigitCaps, self).__init__(**kwargs)
        self.num_capsule = num_capsule    # 4 kelas
        self.dim_capsule = dim_capsule    # 16 dimensi
        self.routings = routings          # r=3

    def build(self, input_shape):
        # input_shape: [Batch_Size, 648, 8]
        self.input_num_capsule = input_shape[1] # 648
        self.input_dim_capsule = input_shape[2] # 8

        # Inisialisasi Matriks Bobot W_ij berukuran [648, 4, 16, 8]
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            name='W'
        )
        self.built = True

    def call(self, inputs):
        # inputs shape: [Batch, 648, 8]
        # Ekspansi dimensi inputs menjadi: [Batch, 648, 1, 8, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), -1)
        
        # Ekspansi matriks W agar memiliki dimensi batch: [1, 648, 4, 16, 8]
        W_expand = tf.expand_dims(self.W, 0)
        
        # Proses Affine Transformation via Einstein Summation (einsum) agar dimensi dijamin presisi
        # Mengalikan W [batch, 648, 4, 16, 8] dengan inputs [batch, 648, 1, 8, 1] 
        # Menghasilkan u_hat dengan shape: [Batch, 648, 4, 16]
        u_hat = tf.einsum('bcijk,bchkl->bcij', tf.tile(W_expand, [tf.shape(inputs)[0], 1, 1, 1, 1]), tf.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1]))

        # Inisialisasi b_ij = 0 dengan shape: [Batch, 648, 4]
        b = tf.zeros(shape=[tf.shape(u_hat)[0], self.input_num_capsule, self.num_capsule])

        # Dynamic Routing Loop (r=3)
        for r in range(self.routings):
            # c shape: [Batch, 648, 4]
            c = tf.nn.softmax(b, axis=-1)
            
            # Menghitung s_j menggunakan einsum: mengalikan koefisien c dengan u_hat
            # Menghasilkan s_j shape: [Batch, 4, 16]
            s_j = tf.einsum('bci,bcij->bij', c, u_hat)
            
            # Aktivasi Squashing -> v_j shape: [Batch, 4, 16]
            v_j = Squash()(s_j)

            # Update nilai b_ij sebelum iterasi terakhir
            if r < self.routings - 1:
                # Dot product antara u_hat [Batch, 648, 4, 16] dan v_j [Batch, 4, 16]
                # Menghasilkan nilai koreksi b shape: [Batch, 648, 4]
                v_j_expand = tf.expand_dims(v_j, 1) # [Batch, 1, 4, 16]
                u_v_dot = tf.reduce_sum(u_hat * v_j_expand, axis=-1)
                b += u_v_dot

        return v_j

    def compute_output_shape(self, input_shape):
        # Membantu Keras melakukan inferensi bentuk output secara eksplisit
        return (input_shape[0], self.num_capsule, self.dim_capsule)