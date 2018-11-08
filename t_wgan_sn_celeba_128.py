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
z_dim = 100


def imread(f):
    x = misc.imread(f)
    x = misc.imresize(x, (img_dim, img_dim))
    return x.astype(np.float32) / 255 * 2 - 1


def data_generator(batch_size=32):
    X = []
    while True:
        np.random.shuffle(imgs)
        for f in imgs:
            X.append(imread(f))
            if len(X) == batch_size:
                X = np.array(X)
                yield X
                X = []


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

x = Conv2D(img_dim,
           (5, 5),
           strides=(2, 2),
           padding='same',
           kernel_constraint=spectral_normalization)(x)
x = LeakyReLU()(x)

for i in range(3):
    x = Conv2D(img_dim * 2**(i + 1),
               (5, 5),
               strides=(2, 2),
               padding='same',
               kernel_constraint=spectral_normalization)(x)
    x = BatchNormalization(gamma_constraint=spectral_normalization)(x)
    x = LeakyReLU()(x)

x = GlobalAveragePooling2D()(x)

e_model = Model(x_in, x)
e_model.summary()


# 判别器
z_in = Input(shape=(K.int_shape(x)[-1],))
z = z_in

z = Dense(512, kernel_constraint=spectral_normalization)(z)
z = LeakyReLU()(z)
z = Dense(1, use_bias=False,
          kernel_constraint=spectral_normalization)(z)

d_model = Model(z_in, z)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(4 * 4 * img_dim * 8)(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((4, 4, img_dim * 8))(z)

for i in range(4):
    z = Conv2DTranspose(img_dim * 4 // 2**i,
                        (5, 5),
                        strides=(2, 2),
                        padding='same')(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

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
x_real_fake = Subtract()([x_real_encoded, x_fake_encoded])
x_fake_real = Subtract()([x_fake_encoded, x_real_encoded])
x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

d_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

d_loss = K.mean(- (x_real_fake_score - x_fake_real_score))
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

g_loss = K.mean(- (x_fake_real_score - x_real_fake_score))
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
