#! -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
import os


if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('../CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 128
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3


def imread(f):
    x = misc.imread(f)
    x = misc.imresize(x, (img_dim, img_dim))
    return x.astype(np.float32) / 255 * 2 - 1


def data_generator(batch_size=64):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X = []


class SpectralNormalization:
    """层的一个包装，用来加上SN。
    """

    def __init__(self, layer, **kwargs):
        self.layer = layer

    def spectral_norm(self, w, r=5):
        w_shape = K.int_shape(w)
        in_dim = np.prod(w_shape[:-1]).astype(int)
        out_dim = w_shape[-1]
        w = K.reshape(w, (in_dim, out_dim))
        u = K.ones((1, in_dim))
        for i in range(r):
            v = K.l2_normalize(K.dot(u, w))
            u = K.l2_normalize(K.dot(v, K.transpose(w)))
        return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

    def spectral_normalization(self, w):
        return w / self.spectral_norm(w)

    def __call__(self, inputs):
        if not self.layer.built:
            input_shape = K.int_shape(inputs)
            self.layer.build(input_shape)
            self.layer.built = True
        if not hasattr(self.layer, 'spectral_normalization'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalization(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalization(self.layer.gamma)
            self.layer.spectral_normalization = True
        return self.layer(inputs)


class Attention(Layer):
    """Attention层
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=[1],
                                     initializer='zeros',
                                     trainable=True)

    def call(self, inputs):
        q_in, k_in, v_in = inputs
        q_shape, k_shape, v_shape = K.shape(q_in), K.shape(k_in), K.shape(v_in)
        q_in = K.reshape(q_in, (q_shape[0], -1, q_shape[-1]))
        k_in = K.reshape(k_in, (k_shape[0], -1, k_shape[-1]))
        v_in = K.reshape(v_in, (v_shape[0], -1, v_shape[-1]))
        attention = K.batch_dot(q_in, k_in, [2, 2])
        attention = K.softmax(attention)
        output = K.batch_dot(attention, v_in, [2, 1])
        output = K.reshape(output, (q_shape[0], q_shape[1], q_shape[2], v_shape[-1]))
        return output * self.gamma

    def compute_output_shape(self, input_shape):
        q_shape, k_shape, v_shape = input_shape
        return q_shape[:3] + (v_shape[-1],)


# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in
num_channels = 64

x = ZeroPadding2D()(x)
x = SpectralNormalization(
    Conv2D(num_channels,
           (4, 4),
           strides=(2, 2),
           use_bias=False))(x)
x = LeakyReLU(0.2)(x)

for i in range(num_layers // 2):
    x = ZeroPadding2D()(x)
    x = SpectralNormalization(
        Conv2D(num_channels * 2,
               (4, 4),
               strides=(2, 2),
               use_bias=False))(x)
    x = SpectralNormalization(
        BatchNormalization())(x)
    x = LeakyReLU(0.2)(x)
    num_channels *= 2

xq = SpectralNormalization(
    Conv2D(num_channels // 8, (1, 1)))(x)
xk = SpectralNormalization(
    Conv2D(num_channels // 8, (1, 1)))(x)
xv = SpectralNormalization(
    Conv2D(num_channels, (1, 1)))(x)
xa = Attention()([xq, xk, xv])
x = Add()([x, xa])

for i in range(num_layers // 2, num_layers):
    x = ZeroPadding2D()(x)
    x = SpectralNormalization(
        Conv2D(num_channels * 2,
               (4, 4),
               strides=(2, 2),
               use_bias=False))(x)
    x = SpectralNormalization(
        BatchNormalization())(x)
    x = LeakyReLU(0.2)(x)
    num_channels *= 2

x = GlobalAveragePooling2D()(x)

e_model = Model(x_in, x)
e_model.summary()


# 判别器
z_in = Input(shape=(K.int_shape(x)[-1],))
z = z_in

z = SpectralNormalization(
    Dense(512))(z)
z = LeakyReLU(0.2)(z)
z = SpectralNormalization(
    Dense(1, use_bias=False,
          activation='sigmoid'))(z)

d_model = Model(z_in, z)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in
num_channels = img_dim * 8 # 固定1024好还是目前这样好？

z = Reshape((1, 1, z_dim))(z)
z = SpectralNormalization(
    Conv2DTranspose(num_channels,
                    (4, 4),
                    padding='valid',
                    use_bias=False))(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)

for i in range(num_layers // 2):
    z = UpSampling2D()(z)
    z = ZeroPadding2D()(z)
    z = SpectralNormalization(
        Conv2D(num_channels // 2, (3, 3)))(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    num_channels //= 2

zq = SpectralNormalization(
    Conv2D(num_channels // 8, (1, 1)))(z)
zk = SpectralNormalization(
    Conv2D(num_channels // 8, (1, 1)))(z)
zv = SpectralNormalization(
    Conv2D(num_channels, (1, 1)))(z)
za = Attention()([zq, zk, zv])
z = Add()([z, za])

for i in range(num_layers // 2, num_layers):
    z = UpSampling2D()(z)
    z = ZeroPadding2D()(z)
    z = SpectralNormalization(
        Conv2D(num_channels // 2, (3, 3)))(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    num_layers = num_layers // 2

z = UpSampling2D()(z)
z = ZeroPadding2D()(z)
z = SpectralNormalization(
    Conv2D(3, (3, 3)))(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_fake = g_model(z_in)
x_real_encoded = e_model(x_in)
x_fake_encoded = e_model(x_fake)
x_real_fake = Subtract()([x_real_encoded, x_fake_encoded])
x_fake_real = Subtract()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

d_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

d_loss = K.mean(- K.log(x_real_fake_score + 1e-9) - K.log(1 - x_fake_real_score + 1e-9))
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False
e_model.trainable = False

x_fake = g_model(z_in)
x_real_encoded = e_model(x_in)
x_fake_encoded = e_model(x_fake)
x_real_fake = Subtract()([x_real_encoded, x_fake_encoded])
x_fake_real = Subtract()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

g_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

g_loss = K.mean(- K.log(1 - x_real_fake_score + 1e-9) - K.log(x_fake_real_score + 1e-9))
g_train_model.add_loss(g_loss)
g_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 检查模型结构
d_train_model.summary()
g_train_model.summary()


# 采样函数
def sample(path):
    n = 9
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            z_sample = np.random.randn(1, z_dim)
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype(int)
    imageio.imwrite(path, figure)


iters_per_sample = 100
total_iter = 1000000
batch_size = 64
img_generator = data_generator(batch_size)

for i in range(total_iter):
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        d_loss = d_train_model.train_on_batch(
            [img_generator.next(), z_sample], None)
    for j in range(2):
        z_sample = np.random.randn(batch_size, z_dim)
        g_loss = g_train_model.train_on_batch(
            [img_generator.next(), z_sample], None)
    if i % 10 == 0:
        print 'iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss)
    if i % iters_per_sample == 0:
        sample('samples/test_%s.png' % i)
    g_train_model.save_weights('./g_train_model.weights')
