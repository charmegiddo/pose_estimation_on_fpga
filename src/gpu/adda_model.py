import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import backend as K
from tensorflow.keras import activations, initializers
from tensorflow.keras.utils import to_categorical#np_utils
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Lambda
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from tensorflow.keras.regularizers import l2
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '/workspace/91_common/'))
import random
import math

#################################################################
## Fields,
#################################################################
width = int(1636 * 0.5)
height = int(1220 * 0.5)
channels = 3
input_shape = (height, width, channels)
classes = 7 # nx, ny, nz, z-nw
act = 'relu'
DropOutRate = 0.15

#################################################################
## Model,
#################################################################
def build_model():
  act_local = act
  input_img=Input(shape=input_shape)
  net = Conv2D(filters=16,
               kernel_size=(2, 2),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(input_img)
               
  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)
  
  net = Dropout(DropOutRate)(net)

  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Dropout(DropOutRate)(net)

  net = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Dropout(DropOutRate)(net)

  net = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Dropout(DropOutRate)(net)

  net = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)
  
  net = Dropout(DropOutRate)(net)

  net = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)

  net = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1), padding="same",
               use_bias=False,
               kernel_initializer="he_normal")(net)
  
  net = Flatten()(net)
  net = Dense(512, activation=act_local, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
  net = Dropout(DropOutRate)(net)
  net = Dense(512, activation=act_local, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
  net = Dropout(DropOutRate)(net)
  net = Dense(classes, use_bias=True, activation=None,kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
  pose_estimation_model = Model(input_img, net, name='pose_estimation_model')
  return pose_estimation_model


def build_extractor_feature_model():
  act_local = act
  input_img=Input(shape=input_shape)
  net = Conv2D(filters=4,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               kernel_initializer="he_normal")(input_img)
               
  net = Conv2D(filters=4,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), padding="same",
               kernel_initializer="he_normal")(net)
  
  net = Conv2D(filters=4,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal")(net)
  
  net = Conv2D(filters=4,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1),
               padding="same",
               kernel_initializer="he_normal")(net)
  
  net = Conv2D(filters=8,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2), 
               padding="same",
               kernel_initializer="he_normal")(net)
  net = Conv2D(filters=8,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1),
               padding="same",
               kernel_initializer="he_normal")(net)
#  net = Lambda(lambda x: K.dropout(x, DropOutRate, seed=None, noise_shape=(1, 1, 1, 256)))(net)
  net = Dropout(DropOutRate)(net) 

  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal")(net)
  net = Conv2D(filters=16,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1),
               padding="same",
               kernel_initializer="he_normal")(net)
#  net = Lambda(lambda x: K.dropout(x, DropOutRate, seed=None, noise_shape=(1, 1, 1, 512)))(net)
  net = Dropout(DropOutRate)(net)

  net = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal")(net)
  net = Conv2D(filters=32,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1),
               padding="same",
               kernel_initializer="he_normal")(net)
#  net = Lambda(lambda x: K.dropout(x, DropOutRate, seed=None, noise_shape=(1, 1, 1, 1024)))(net)
  net = Dropout(DropOutRate)(net) 
  
  net = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal")(net)
  net = Conv2D(filters=64,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1),
               padding="same",
               kernel_initializer="he_normal")(net)
#  net = Lambda(lambda x: K.dropout(x, DropOutRate, seed=None, noise_shape=(1, 1, 1, 2048)))(net)
  net = Dropout(DropOutRate)(net) 
  
  net = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(2, 2),
               padding="same",
               kernel_initializer="he_normal")(net)
  net = Conv2D(filters=128,
               kernel_size=(3, 3),
               activation=act_local,
               strides=(1, 1),
               padding="same",
               kernel_initializer="he_normal")(net)
#  net = Lambda(lambda x: K.dropout(x, DropOutRate, seed=None, noise_shape=(1, 1, 1, 2048)))(net)
  net = Dropout(DropOutRate)(net)
  net = Flatten()(net)
  extractor_feature_model = Model(input_img, net, name='extractor_feature_model')
  return extractor_feature_model

def build_pose_estimation_model(_input_shape):
   input_flatten=Input(shape=_input_shape)
   act_local = 'tanh'
   net = Dense(256, activation=act_local, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(input_flatten)
   net = Dropout(DropOutRate)(net)
   net = Dense(256, activation=act_local, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='intermediate')(net)
   net = Dropout(DropOutRate)(net)
   net = Dense(256, activation=act_local, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
   net = Dense(classes, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(net)
   
   pose_estimation_model = Model(input_flatten, net, name='pose_estimation_model')
   return pose_estimation_model


def build_discriminator_model(_latent_shape):
  latent_input = Input(shape=_latent_shape, name='discriminator0')
  net = Dense(256, activation=act, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='discriminator1')(latent_input)
  net = Dense(256, activation=act, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='discriminator2')(net)
  net = BatchNormalization()(net)
  #net = Dense(512, activation=act, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', name='discriminator3')(net)
  #net = BatchNormalization()(net)
  output = Dense(2, activation='softmax', name='discriminator_output')(net)
  discriminator_model = Model(inputs=latent_input, outputs=output, name='discriminator')
  return discriminator_model

# model2(model1)
def concat_model(_model1, _model2):
  return Model(_model1.input, _model2(_model1.output))

def view_model(_model):
  for i, l in enumerate(_model.layers):
    print(str(i) + ", " + str(l) + ", "  + l.name + ", " + str(l.trainable))
