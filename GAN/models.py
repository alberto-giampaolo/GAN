import numpy as np

from keras.layers import Input, Dense, Add, Multiply
from keras.layers import Reshape, UpSampling1D, Flatten, concatenate, Cropping1D
from keras.layers import Activation, LeakyReLU, PReLU
from keras.layers import BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam, RMSprop

from copy import copy

from . import plotting 

from .base import Builder, MyGAN

from .ffwd import MyFFGAN
from .conv import MyConvGAN
from .unet import MyUnetGAN
# from .pseudoq import MyPQGAN

from .wgan import WeightClip, wgan_loss

import keras.losses
import keras.constraints

keras.losses.wgan_loss = wgan_loss
keras.constraints.WeightClip = WeightClip

