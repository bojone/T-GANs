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
from keras.datasets import cifar10


if not os.path.exists('samples'):
    os.mkdir('samples')


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255 - 0.5
x_test = x_test.astype('float32') / 255 - 0.5
img_dim = x_train.shape[1]
z_dim = 64


def spectral_norm(w, r=5):
    w_shape = K.int_shape(w)
    in_dim = np.prod(w_shape[:-1]).astype(int)
    out_dim = w_shape[-1]
    w = K.reshape(w, (in_dim, out_dim))
    u = K.ones((1, in_dim))
    for i in range(r):
        v = K.l2_normalize(K.dot(u, w))
        u = K.l2_normalize(K.dot(v, K.transpose(w)))
    return K.sum(K.dot(K.dot(u, w), K.transpose(v)))


def spectral_normalization(w):
    return w / spectral_norm(w)


# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

x = Conv2D(32, (5,5), strides=(1,1),
           kernel_constraint=spectral_normalization)(x)
x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
x = LeakyReLU()(x)
x = Conv2D(64, (4,4), strides=(2,2),
           kernel_constraint=spectral_normalization)(x)
x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
x = LeakyReLU()(x)
x = Conv2D(128, (4,4), strides=(1,1),
           kernel_constraint=spectral_normalization)(x)
x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
x = LeakyReLU()(x)
x = Conv2D(256, (4,4), strides=(2,2),
           kernel_constraint=spectral_normalization)(x)
x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
x = LeakyReLU()(x)

x = Conv2D(z_dim, (1,1), strides=(1,1),
           kernel_constraint=spectral_normalization)(x)

x = GlobalAveragePooling2D()(x)

e_model = Model(x_in, x)
e_model.summary()


# 判别器
z_in = Input(shape=(K.int_shape(x)[-1],))
z = z_in

z = Dense(256, kernel_constraint=spectral_normalization)(z)
z = LeakyReLU(0.1)(z)
z = Dense(1, use_bias=False,
          kernel_constraint=spectral_normalization,
          activation='sigmoid')(z)

d_model = Model(z_in, z)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Reshape((1, 1, z_dim))(z)
z = Conv2DTranspose(256, (4,4), strides=(1,1))(z)
z = BatchNormalization()(z)
z = LeakyReLU(0.1)(z)
z = Conv2DTranspose(128, (4,4), strides=(2,2))(z)
z = BatchNormalization()(z)
z = LeakyReLU(0.1)(z)
z = Conv2DTranspose(64, (4,4), strides=(1,1))(z)
z = BatchNormalization()(z)
z = LeakyReLU(0.1)(z)
z = Conv2DTranspose(32, (4,4), strides=(2,2))(z)
z = BatchNormalization()(z)
z = LeakyReLU(0.1)(z)
z = Conv2DTranspose(32, (5,5), strides=(1,1))(z)
z = BatchNormalization()(z)
z = LeakyReLU(0.1)(z)
z = Conv2D(32, (1,1), strides=(1,1))(z)
z = BatchNormalization()(z)
z = LeakyReLU(0.1)(z)
z = Conv2D(3, (1,1), strides=(1,1), activation='tanh')(z)

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
    n = 10
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


iters_per_sample = 200
total_iter = 1000000
batch_size = 64

for i in range(total_iter):
    for j in range(1):
        z_sample = np.random.randn(batch_size, z_dim)
        x_sample = x_train[np.random.choice(len(x_train), size=batch_size)]
        d_loss = d_train_model.train_on_batch([x_sample, z_sample], None)
    for j in range(2):
        z_sample = np.random.randn(batch_size, z_dim)
        x_sample = x_train[np.random.choice(len(x_train), size=batch_size)]
        g_loss = g_train_model.train_on_batch([x_sample, z_sample], None)
    if i % 10 == 0:
        print 'iter: %s, d_loss: %s, g_loss: %s' % (i, d_loss, g_loss)
    if i % iters_per_sample == 0:
        sample('samples/test_%s.png' % i)
        g_train_model.save_weights('./g_train_model.weights')
