#! -*- coding: utf-8 -*-
# 通过loss偏置来解决sgan的梯度消失问题，可以去掉sn

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
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)


def imread(f):
    x = misc.imread(f, mode='RGB')
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


def log_sigmoid(x):
    return - K.softplus(- x)


# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               use_bias=False,
               padding='same')(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    if i == num_layers // 2:
        xq = Conv2D(num_channels // 8, (1, 1))(x)
        xk = Conv2D(num_channels // 8, (1, 1))(x)
        xv = Conv2D(num_channels, (1, 1))(x)
        xa = Attention()([xq, xk, xv])
        x = Add()([x, xa])

x = GlobalAveragePooling2D()(x)

e_model = Model(x_in, x)
e_model.summary()


# 判别器
z_in = Input(shape=(K.int_shape(x)[-1] * 2,))
z = z_in

z = Dense(512, use_bias=False)(z)
z = LeakyReLU(0.2)(z)
z = Dense(1, use_bias=False)(z)

d_model = Model(z_in, z)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    if i == num_layers // 2 - 1:
        zq = Conv2D(num_channels // 8, (1, 1))(z)
        zk = Conv2D(num_channels // 8, (1, 1))(z)
        zv = Conv2D(num_channels, (1, 1))(z)
        za = Attention()([zq, zk, zv])
        z = Add()([z, za])

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same')(z)
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
x_real_fake = Concatenate()([x_real_encoded, x_fake_encoded])
x_fake_real = Concatenate()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

d_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

d_loss = K.mean(- log_sigmoid(x_real_fake_score) - 0.5 * log_sigmoid(- x_real_fake_score) - 0.5 * log_sigmoid(- x_fake_real_score))
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False
e_model.trainable = False

x_fake = g_model(z_in)
x_real_encoded = e_model(x_in)
x_fake_encoded = e_model(x_fake)
x_real_fake = Concatenate()([x_real_encoded, x_fake_encoded])
x_fake_real = Concatenate()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

g_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

g_loss = K.mean(- log_sigmoid(x_fake_real_score) - 0.5 * log_sigmoid(- x_fake_real_score) - 0.5 * log_sigmoid(- x_real_fake_score))
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
