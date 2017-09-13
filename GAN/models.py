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

