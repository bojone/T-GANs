#! -*- coding: utf-8 -*-

import glob
from tqdm import tqdm
import numpy as np
import scipy as sp
from keras.applications.inception_v3 import InceptionV3,preprocess_input


train = glob.glob('../CelebA-HQ/train/*.png')
valid = glob.glob('../CelebA-HQ/valid/*.png')


def imread(f):
    x = sp.misc.imread(f, mode='RGB')
    x = sp.misc.imresize(x, (299, 299))
    return x


x_real = np.array([imread(f) for f in tqdm(iter(train))]).astype('float32')
x_fake = np.array([imread(f) for f in tqdm(iter(valid))]).astype('float32')


class FID:
    def __init__(self, x_real):
        self.base_model = InceptionV3(include_top=False, pooling='avg')
        self.mu_real,self.sigma_real = self.get_mu_sigma(x_real)
    def get_mu_sigma(self, x):
        x = preprocess_input(x.copy())
        h = self.base_model.predict(x, verbose=True, batch_size=128)
        mu = h.mean(0)
        sigma = np.cov(h.T)
        return mu,sigma
    def distance(self, x_fake):
        mu_real,sigma_real = self.mu_real,self.sigma_real
        mu_fake,sigma_fake = self.get_mu_sigma(x_fake)
        mu_diff = mu_real - mu_fake
        sigma_root = sp.linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)[0]
        sigma_diff = sigma_real + sigma_fake - 2 * sigma_root
        return np.real((mu_diff**2).sum() + np.trace(sigma_diff))


fid = FID(x_real)
fid.distance(x_fake)
