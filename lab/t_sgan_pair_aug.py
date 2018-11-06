#! -*- coding: utf-8 -*-
# 普通的T-SGAN，通过shuffle构造更多的真假对来达到正则效果。
# （样本多了，就不容易过拟合了）

import numpy as np
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.initializers import RandomNormal
from keras.optimizers import Adam
import os


if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('../CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 256
z_dim = 128
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 64


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

    def spectral_normalize(self, w):
        return w / self.spectral_norm(w)

    def __call__(self, inputs):
        if not self.layer.built:
            input_shape = K.int_shape(inputs)
            self.layer.build(input_shape)
            self.layer.built = True
        if not hasattr(self.layer, 'spectral_normalized'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalize(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalize(self.layer.gamma)
            self.layer.spectral_normalized = True
        return self.layer(inputs)


def log_sigmoid(x):
    return - K.softplus(- x)


# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = SpectralNormalization(
        Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               use_bias=False,
               padding='same',
               kernel_initializer=RandomNormal(stddev=0.02)))(x)
    if i > 0:
        x = SpectralNormalization(
            BatchNormalization())(x)
    x = LeakyReLU(0.2)(x)

x = GlobalAveragePooling2D()(x)

e_model = Model(x_in, x)
e_model.summary()


# 判别器
z_in = Input(shape=(K.int_shape(x)[-1],))
z = z_in

z = SpectralNormalization(
    Dense(512, use_bias=False,
          kernel_initializer=RandomNormal(stddev=0.02)))(z)
z = LeakyReLU(0.2)(z)
z = SpectralNormalization(
    Dense(1, use_bias=False,
          kernel_initializer=RandomNormal(stddev=0.02)))(z)

d_model = Model(z_in, z)
d_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels,
          kernel_initializer=RandomNormal(stddev=0.02))(z)
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Reshape((f_size, f_size, max_num_channels))(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=RandomNormal(stddev=0.02))(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=RandomNormal(stddev=0.02))(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# shuffle层，打乱第一个轴
def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = K.tf.random_shuffle(idxs)
    return K.gather(x, idxs)


# 整合模型（训练判别器）
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))
g_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)
x_real_encoded = e_model(x_real)
x_fake_encoded = e_model(x_fake)
x_fake_encodeds = [x_fake_encoded]

for i in range(2):
    _ = Lambda(shuffling)(x_fake_encoded)
    x_fake_encodeds.append(_)

x_real_fake,x_fake_real = [],[]

for fe in x_fake_encodeds:
    x_real_fake.append(Subtract()([x_real_encoded, fe]))
    x_fake_real.append(Subtract()([fe, x_real_encoded]))

x_real_fake = Concatenate(0)(x_real_fake)
x_fake_real = Concatenate(0)(x_fake_real)

x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

d_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

d_loss = K.mean(- log_sigmoid(x_real_fake_score- x_fake_real_score))
d_train_model.add_loss(d_loss)
d_train_model.compile(optimizer=Adam(2e-4, 0.5))


# 整合模型（训练生成器）
g_model.trainable = True
d_model.trainable = False
e_model.trainable = False

x_real = x_in
x_fake = g_model(z_in)
x_real_encoded = e_model(x_real)
x_fake_encoded = e_model(x_fake)
x_fake_encodeds = [x_fake_encoded]

for i in range(2):
    _ = Lambda(shuffling)(x_fake_encoded)
    x_fake_encodeds.append(_)

x_real_fake,x_fake_real = [],[]

for fe in x_fake_encodeds:
    x_real_fake.append(Subtract()([x_real_encoded, fe]))
    x_fake_real.append(Subtract()([fe, x_real_encoded]))

x_real_fake = Concatenate(0)(x_real_fake)
x_fake_real = Concatenate(0)(x_fake_real)

x_real_fake_score = d_model(x_real_fake)
x_fake_real_score = d_model(x_fake_real)

g_train_model = Model([x_in, z_in],
                      [x_real_fake_score, x_fake_real_score])

g_loss = K.mean(- log_sigmoid(x_fake_real_score - x_real_fake_score))
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
