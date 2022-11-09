import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras.models import load_model, Model, clone_model
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
import cm_dataset
#import cm_visualized_grad as vg
#import sepres_drop as qn
import random
import math
import adda_model as am
import adda_utils as au
import copy as cp
import gc

# new
src_train = False
src_prediction = False


#
src_train = True
#src_prediction = True

if __name__ == '__main__':
  input_shape = am.input_shape
  classes = am.classes
  prefixsub = "_30-70m_"
  src_ds = cm_dataset.Dataset(classes=classes, input_shape=input_shape, dataPath='/workspace/dataset/' + prefixsub + '135000img/') 
  sm  = au.Save_model()
  norm_range = [(-14, 14),(-10.4, 10.4),(30.0, 70.0),(-1, 1),(-1, 1),(-1, 1),(-180, 180)] # 30-70m
  if src_train:
    # build network
    extractor_feature_model = am.build_extractor_feature_model()
    pose_estimation_model = am.build_pose_estimation_model(_input_shape=extractor_feature_model.output.shape[1:])
    trainer_model = am.concat_model(extractor_feature_model, pose_estimation_model)
    opt = keras.optimizers.Adam(lr=1e-4) 
    trainer_model.compile(loss=au.dloss_function, optimizer=opt)
    print(trainer_model.summary())
    # field
    train_gepochs = 5000
    train_epochs = 10
    train_batch_num = 150
    train_mini_batch_num = 10
    prev_error = 9e+9
    src_test_X, src_test_Y = src_ds.load_debris_rocket(_num=300, _norm_range=norm_range, filename="test.log", randomly=False)
    src_test_filename = src_ds.get_file_list()
    print("Start Training")
    # training
    for i in range(train_gepochs):
      print(str(i) + " epochs")
      # training
      gc.collect() # memory clear
      src_X, src_Y = src_ds.load_debris_rocket(_num=train_batch_num, _norm_range=norm_range, filename="train.log", randomly=True)
      trainer_model.fit(src_X, src_Y, batch_size=train_mini_batch_num, epochs=train_epochs, verbose=1)
      if i % 5 == 0:
        error_losses = au.prediction_mcdo(trainer_model, src_ds, src_test_X, src_test_Y, norm_range, src_test_filename, 20)
        now_error = error_losses
        if now_error < prev_error:
          print("[SAVE EVENT] Update Minimum Error: " + str(prev_error) + " -> " + str(now_error))
          prev_error = now_error
          trainer_model.save_weights('./weights/' + prefixsub + 'src_full_estimation_' + str(prev_error) + '.hdf5')
          extractor_feature_model.save_weights('./weights/' + prefixsub + 'src_extractor_feature_model_' + str(prev_error) + '.hdf5')
          pose_estimation_model.save_weights('./weights/' + prefixsub + 'src_pose_estimation_model_' + str(prev_error) + '.hdf5')
          trainer_model.save_weights('./weights/' + prefixsub + 'src_latest.hdf5')
          trainer_model.save('./weights/' + prefixsub + 'for_fpga.hdf5')

  if src_prediction:
    # build network
    extractor_feature_model = am.build_extractor_feature_model()
    pose_estimation_model = am.build_pose_estimation_model(_input_shape=extractor_feature_model.output.shape[1:])
    prediction_model = am.concat_model(extractor_feature_model, pose_estimation_model)
    opt = keras.optimizers.Adam(lr=1e-4) 
    prediction_model.compile(loss=au.dloss_function, optimizer=opt)
    print(prediction_model.summary())
    # change your dataset
    tmp_ds = src_ds
    tmp_X, tmp_Y = src_ds.load_debris_rocket(_num=1000, _norm_range=norm_range, filename="test.log", randomly=False)
    tmp_filename = tmp_ds.get_file_list()
    prediction_model.load_weights('./weights/' + prefixsub  + 'src_full_estimation_182.28001239506403.hdf5')
    print("Start Testing")
    # testing
    error_losses = au.prediction_mcdo(prediction_model, tmp_ds, tmp_X, tmp_Y, norm_range, tmp_filename, 120)

