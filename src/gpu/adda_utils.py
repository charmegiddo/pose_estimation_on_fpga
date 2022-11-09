import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy as cp
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
PI = 3.14159265358979323846264338327950288 
PI_DEGREE = 180.0
RAD2DEG = 180.0 / PI  # 57.2957795
DEG2RAD = PI  / 180.0 # 0.01745329

def ae_loss_function(y, y_hat):
  sim = Lambda(lambda x: x[0, :, :])(y_hat)
  real = Lambda(lambda x: x[1, :, :])(y_hat)
  return 0

def dloss_function(y, y_hat):
    e1xyz = Lambda(lambda x: x[:, 0:3])(y)
    e2xyz = Lambda(lambda x: x[:, 0:3])(y_hat)
    error_euclidXyz = K.sum(K.pow(e2xyz - e1xyz, 2))
    # target
    q1xyz = Lambda(lambda x: x[:, 3:6])(y_hat)
    q1w = Lambda(lambda x:   x[:, 6])(y_hat)
    # opposite
    q2xyz = Lambda(lambda x: x[:, 3:6])(y)
    q2w = Lambda(lambda x: x[:, 6])(y)
    # nxnynz
    error_xyz = K.sum(K.pow(q2xyz - q1xyz, 2))
    # error_w = K.sum(K.pow(K.pow(q1w,2) - K.pow(q2w,2), 2))
    q1w = K.clip(q1w, -1.0, 1.0) # -1 ~ +1
    q2w = K.clip(q2w, -1.0, 1.0) # -1 ~ +1
    q1w = (q1w * PI_DEGREE) * DEG2RAD #  -1 ~ +1 -> -180 ~ +180 -> -3.14 ~ +3.14 deg2rad (pi /180) 
    q2w = (q2w * PI_DEGREE) * DEG2RAD #  -1 ~ +1 -> -180 ~ +180 -> -3.14 ~ +3.14 deg2rad (pi /180) 
    diff_rad = tf.abs( tf.atan2( tf.sin(q1w - q2w), tf.cos(q1w - q2w)) ) #diff = tf.math.acos(K.cos(q1w) * K.cos(q2w) + K.sin(q1w) * K.sin(q2w)) * 57.29577951308232 # 180 / pi
    ## norm
    diff = (diff_rad * RAD2DEG) / PI_DEGREE # rad -> deg -> 0.0 ~ +1.0
    error_w = K.sum(diff)
    return  error_euclidXyz + error_xyz# + (error_w*2)

def normalization(_prd):
  for i in range(len(_prd)):
    pred = _prd[i]
    leng = np.sqrt((pred[3]**2) + (pred[4]**2) + (pred[5]**2))
    pred[3] /= leng
    pred[4] /= leng
    pred[5] /= leng
  return _prd
  

def prediction_mcdo2(_result, _src_ds, _test_X, _test_Y, _norm_range, _test_filename, _trial):
  ds = _src_ds
  test_X = _test_X
  test_Y = _test_Y
  test_filename = _test_filename
  trial = _trial
  infs = np.asarray(normalization(_result))
  s_minx, s_maxx = _norm_range[0]
  s_miny, s_maxy = _norm_range[1]
  s_minz, s_maxz = _norm_range[2]
  s_minnx, s_maxnx = _norm_range[3]
  s_minny, s_maxny = _norm_range[4]
  s_minnz, s_maxnz = _norm_range[5]
  s_minrw, s_maxrw = _norm_range[6]
  print("---------------------Uncertainty Error----------------------")
  #infs = np.zeros((trial, len(test_X), 7)) # trail, data_num, output_ parameter
  # get mean = inference
  e1x = np.zeros((len(test_Y)),dtype=np.float)
  e1y = np.zeros((len(test_Y)),dtype=np.float)
  e1z = np.zeros((len(test_Y)),dtype=np.float)
  q1x = np.zeros((len(test_Y)),dtype=np.float)
  q1y = np.zeros((len(test_Y)),dtype=np.float)
  q1z = np.zeros((len(test_Y)),dtype=np.float)
  q1w = np.zeros((len(test_Y)),dtype=np.float)
  q1w_sin = np.zeros((len(test_Y)),dtype=np.float)
  q1w_cos = np.zeros((len(test_Y)),dtype=np.float)
  p_q1w_sin = np.zeros((len(test_Y), len(infs)),dtype=np.float)
  p_q1w_cos = np.zeros((len(test_Y), len(infs)),dtype=np.float)
  p_q1w_theta = np.zeros((len(test_Y), len(infs)),dtype=np.float)

  e2x = np.zeros((len(test_Y)),dtype=np.float)
  e2y = np.zeros((len(test_Y)),dtype=np.float)
  e2z = np.zeros((len(test_Y)),dtype=np.float)
  q2x = np.zeros((len(test_Y)),dtype=np.float)
  q2y = np.zeros((len(test_Y)),dtype=np.float)
  q2z = np.zeros((len(test_Y)),dtype=np.float)
  q2w = np.zeros((len(test_Y)),dtype=np.float)

  prd = infs
  for c in range(len(test_Y)):
    e1x[c] = prd[c, 0]
    e1y[c] = prd[c, 1]
    e1z[c] = prd[c, 2]
    q1x[c] = prd[c, 3]
    q1y[c] = prd[c, 4]
    q1z[c] = prd[c, 5]
    q1w[c] = (np.clip(prd[c, 6], -1.0, 1.0) ) # -1.0 ~ +1.0

    # -1 ~ +1 -> degree -> radian
    q1w_sin[c] = np.sin((np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD) # -1.0 ~ +1.0 -> -180 ~ +180 -> -3.1415 ~ +3.1415
    q1w_cos[c] = np.cos((np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD) # -1.0 ~ +1.0 -> -180 ~ +180 -> -3.1415 ~ +3.1415

    e2x[c] = test_Y[c, 0] # x
    e2y[c] = test_Y[c, 1] # y
    e2z[c] = test_Y[c, 2] # z
    q2x[c] = test_Y[c, 3] # vx
    q2y[c] = test_Y[c, 4] # vy
    q2z[c] = test_Y[c, 5] # vz
    q2w[c] = test_Y[c, 6] # w around axis-z

  e1x_mean = e1x
  e1y_mean = e1y
  e1z_mean = e1z
  q1x_mean = q1x
  q1y_mean = q1y
  q1z_mean = q1z
  q1w_mean =  q1w
  q1w_sin_mean = q1w_sin
  q1w_cos_mean = q1w_cos

  # get var
  e1x_s = np.zeros((len(test_Y)),dtype=np.float)
  e1y_s = np.zeros((len(test_Y)),dtype=np.float)
  e1z_s = np.zeros((len(test_Y)),dtype=np.float)
  q1x_s = np.zeros((len(test_Y)),dtype=np.float)
  q1y_s = np.zeros((len(test_Y)),dtype=np.float)
  q1z_s = np.zeros((len(test_Y)),dtype=np.float)
  q1w_s = np.zeros((len(test_Y)),dtype=np.float)

  # get loss
  lex = np.abs(e1x_mean - e2x)
  ley = np.abs(e1y_mean - e2y)
  lez = np.abs(e1z_mean - e2z)
  lx = np.abs(q1x_mean - q2x)
  ly = np.abs(q1y_mean - q2y)
  lz = np.abs(q1z_mean - q2z)
  A =  (q1w_mean * PI_DEGREE) * DEG2RAD
  B =  (q2w * PI_DEGREE) * DEG2RAD
  lw = np.abs(np.arctan2(np.sin(A - B), np.cos(A - B)))  * RAD2DEG # rad -> degree

  error_sums = 0
  error_losses = 0
  error_losses_xyz = 0
  error_losses_vxvyvz = 0
  error_losses_w = 0
  print("a: ground truth, p: predict, l: loss, s: variance; w loss is degree unit")
  for c in range(len(test_Y)):
    # var
    error_sum = e1x_s[c] + e1y_s[c] + e1z_s[c] + q1x_s[c] + q1y_s[c] + q1z_s[c] + ((q1w_s[c]/PI_DEGREE)*2) # degree -> 0~2
    error_sums += error_sum
    # total error
    error_loss = lex[c] + ley[c] + lez[c] + lx[c] + ly[c] + lz[c]# + ((lw[c]/PI_DEGREE)*2)
    #error_loss = (lw[c]/PI_DEGREE)*2
    error_losses += error_loss

    # each error
    error_losses_xyz += (lex[c] + ley[c] + lez[c])
    error_losses_vxvyvz += (lx[c] + ly[c] + lz[c])
    error_losses_w += ((lw[c]/PI_DEGREE)*2) # loss, 0 ~ +2

    e1x_scale = ds.min_max_decode_n1_1(e1x_mean[c], s_minx, s_maxx)
    e1y_scale = ds.min_max_decode_n1_1(e1y_mean[c], s_miny, s_maxy)
    e1z_scale = ds.min_max_decode_n1_1(e1z_mean[c], s_minz, s_maxz)
    q1x_scale = ds.min_max_decode_n1_1(q1x_mean[c], s_minnx, s_maxnx)
    q1y_scale = ds.min_max_decode_n1_1(q1y_mean[c], s_minny, s_maxny)
    q1z_scale = ds.min_max_decode_n1_1(q1z_mean[c], s_minnz, s_maxnz)
    # todo: w_scale

    e2x_scale = ds.min_max_decode_n1_1(e2x[c],      s_minx, s_maxx)
    e2y_scale = ds.min_max_decode_n1_1(e2y[c],      s_miny, s_maxy)
    e2z_scale = ds.min_max_decode_n1_1(e2z[c],      s_minz, s_maxz)
    q2x_scale = ds.min_max_decode_n1_1(q2x[c],      s_minnx, s_maxnx)
    q2y_scale = ds.min_max_decode_n1_1(q2y[c],      s_minny, s_maxny)
    q2z_scale = ds.min_max_decode_n1_1(q2z[c],      s_minnz, s_maxnz)

    w_var = 1 - (q1w_sin_mean[c] ** 2 + q1w_cos_mean[c] ** 2)
    print( " |%s| %s, sum_var[%6.4f], sum_loss[%6.4f],\n\
            x[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            y[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            z[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \n\
            vx[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            vy[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            vz[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            w[a:%6.4f (%6.4f), p:%6.4f (%6.4f), v: %6.4f, l:%6.4f, s:%6.4f]" % (
        str(c).zfill(4), test_filename[c], error_sum, error_loss,
        e2x[c], e2x_scale, e1x_mean[c], e1x_scale, lex[c], e1x_s[c],
        e2y[c], e2y_scale, e1y_mean[c], e1y_scale, ley[c], e1y_s[c],
        e2z[c], e2z_scale, e1z_mean[c], e1z_scale, lez[c], e1z_s[c],
        q2x[c], q2x_scale, q1x_mean[c], q1x_scale, lx[c], q1x_s[c],
        q2y[c], q2y_scale, q1y_mean[c], q1y_scale, ly[c], q1y_s[c],
        q2z[c], q2z_scale, q1z_mean[c], q1z_scale, lz[c], q1z_s[c],
        q2w[c], q2w[c] * PI_DEGREE, q1w_mean[c], q1w_mean[c] * PI_DEGREE, w_var, lw[c], q1w_s[c]))

  print("---------------------------------------------------CSV---------------------------------------------------")
  print( "num, filename, sum_var, sum_loss, x[a], x[a_scale], x[p], x[p_scale], x[l], x[s], y[a], y[a_scale], y[p], y[p_scale], y[l], y[s], z[a], z[a_scale], z[p], z[p_scale], z[l], z[s], vx[a],vx[a_scale], vx[p], vx[p_scale], vx[l], vx[s], vy[a], vy[a_scale], vy[p], vy[p_scale], vy[l], vy[s], vz[a], vz[a_scale], vz[p], vz[p_scale], vz[l], vz[s], w[a], w[a-deg], w[p], w[p-deg], w[v], w[l], w[s]" )
  for c in range(len(test_Y)):
    # var
    error_sum = e1x_s[c] + e1y_s[c] + e1z_s[c] + q1x_s[c] + q1y_s[c] + q1z_s[c] + ((q1w_s[c]/PI_DEGREE)*2) # degree -> 0~2
    # loss
    error_loss = lex[c] + ley[c] + lez[c] + lx[c] + ly[c] + lz[c] #+ ((lw[c]/PI_DEGREE)*2)

    # scale
    e1x_scale = ds.min_max_decode_n1_1(e1x_mean[c], s_minx, s_maxx)
    e1y_scale = ds.min_max_decode_n1_1(e1y_mean[c], s_miny, s_maxy)
    e1z_scale = ds.min_max_decode_n1_1(e1z_mean[c], s_minz, s_maxz)
    q1x_scale = ds.min_max_decode_n1_1(q1x_mean[c], s_minnx, s_maxnx)
    q1y_scale = ds.min_max_decode_n1_1(q1y_mean[c], s_minny, s_maxny)
    q1z_scale = ds.min_max_decode_n1_1(q1z_mean[c], s_minnz, s_maxnz)

    e2x_scale = ds.min_max_decode_n1_1(e2x[c],      s_minx, s_maxx) 
    e2y_scale = ds.min_max_decode_n1_1(e2y[c],      s_miny, s_maxy)
    e2z_scale = ds.min_max_decode_n1_1(e2z[c],      s_minz, s_maxz)
    q2x_scale = ds.min_max_decode_n1_1(q2x[c],      s_minnx, s_maxnx)
    q2y_scale = ds.min_max_decode_n1_1(q2y[c],      s_minny, s_maxny)
    q2z_scale = ds.min_max_decode_n1_1(q2z[c],      s_minnz, s_maxnz)

    w_var = 1 - (q1w_sin_mean[c] ** 2 + q1w_cos_mean[c] ** 2)
    print( "%s, %s, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f" % (
        str(c).zfill(4), test_filename[c], error_sum, error_loss, 
        e2x[c], e2x_scale, e1x_mean[c], e1x_scale, lex[c], e1x_s[c],
        e2y[c], e2y_scale, e1y_mean[c], e1y_scale, ley[c], e1y_s[c],
        e2z[c], e2z_scale, e1z_mean[c], e1z_scale, lez[c], e1z_s[c],
        q2x[c], q2x_scale, q1x_mean[c], q1x_scale, lx[c] , q1x_s[c],
        q2y[c], q2y_scale, q1y_mean[c], q1y_scale, ly[c] , q1y_s[c],
        q2z[c], q2z_scale, q1z_mean[c], q1z_scale, lz[c] , q1z_s[c],
        q2w[c], q2w[c] * PI_DEGREE, q1w_mean[c], q1w_mean[c] * PI_DEGREE, w_var, lw[c], q1w_s[c]))

  print("errors (var): %f" % (error_sums))
  print("errors (loss): %f, xyz: %f, vxvyvz: %f, w: %f" % (error_losses, error_losses_xyz, error_losses_vxvyvz, error_losses_w))
  return error_losses


def prediction_mcdo(_model, _src_ds, _test_X, _test_Y, _norm_range, _test_filename, _trial):
  model = _model
  ds = _src_ds
  test_X = _test_X
  test_Y = _test_Y
  test_filename = _test_filename
  trial = _trial
  prd = model.predict(test_X)
  s_minx, s_maxx = _norm_range[0]
  s_miny, s_maxy = _norm_range[1]
  s_minz, s_maxz = _norm_range[2]
  s_minnx, s_maxnx = _norm_range[3]
  s_minny, s_maxny = _norm_range[4]
  s_minnz, s_maxnz = _norm_range[5]
  s_minrw, s_maxrw = _norm_range[6]
  print("---------------------Uncertainty Error----------------------")
  divided_num = 64
  infs = np.zeros((trial, len(test_X), 7)) # trail, data_num, output_ parameter

  # separate inference
  if(len(test_X) > divided_num):
    split_num = int( (len(test_X) / divided_num) - 0.01 )
    for _split in range(split_num):
      # before once from last
      start_idx = _split * divided_num
      end_idx = (_split + 1) * divided_num
      print("[done]", start_idx, ":" ,end_idx)
      tmp_infs = []
      for i in range (trial):
        tmp_infs.append(model.predict(test_X[start_idx:end_idx]))
      infs[:, start_idx:end_idx, :] = cp.deepcopy(np.asarray(tmp_infs))
    # last
    start_idx = (split_num * divided_num)
    print("[done]", start_idx, ":")
    tmp_infs = []
    for i in range (trial):
      tmp_infs.append(model.predict(test_X[start_idx:]))
    infs[:, start_idx:, :] = cp.deepcopy(np.asarray(tmp_infs))
  else:
    tmp_infs = []
    for i in range (trial):
      tmp_infs.append(model.predict(test_X))
    infs[:, :, :] = cp.deepcopy(np.asarray(tmp_infs))
  
  # in the case of over 128 data, this programs could not process 
  # infs = []
  #for i in range (trial):
  #  infs.append(model.predict(test_X))

  # get mean = inference
  e1x = np.zeros((len(test_Y)),dtype=np.float)
  e1y = np.zeros((len(test_Y)),dtype=np.float)
  e1z = np.zeros((len(test_Y)),dtype=np.float)
  q1x = np.zeros((len(test_Y)),dtype=np.float)
  q1y = np.zeros((len(test_Y)),dtype=np.float)
  q1z = np.zeros((len(test_Y)),dtype=np.float)
  q1w = np.zeros((len(test_Y)),dtype=np.float)
  q1w_sin = np.zeros((len(test_Y)),dtype=np.float)
  q1w_cos = np.zeros((len(test_Y)),dtype=np.float)
  p_q1w_sin = np.zeros((len(test_Y), len(infs)),dtype=np.float)
  p_q1w_cos = np.zeros((len(test_Y), len(infs)),dtype=np.float)
  p_q1w_theta = np.zeros((len(test_Y), len(infs)),dtype=np.float)

  e2x = np.zeros((len(test_Y)),dtype=np.float)
  e2y = np.zeros((len(test_Y)),dtype=np.float)
  e2z = np.zeros((len(test_Y)),dtype=np.float)
  q2x = np.zeros((len(test_Y)),dtype=np.float)
  q2y = np.zeros((len(test_Y)),dtype=np.float)
  q2z = np.zeros((len(test_Y)),dtype=np.float)
  q2w = np.zeros((len(test_Y)),dtype=np.float)

  for i in range (len(infs)):
    prd = infs[i]
    for c in range(len(test_Y)):
      e1x[c] += prd[c, 0]
      e1y[c] += prd[c, 1]
      e1z[c] += prd[c, 2]
      q1x[c] += prd[c, 3]
      q1y[c] += prd[c, 4]
      q1z[c] += prd[c, 5]
      q1w[c] += (np.clip(prd[c, 6], -1.0, 1.0) ) # -1.0 ~ +1.0

      # -1 ~ +1 -> degree -> radian
      q1w_sin[c] += np.sin((np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD) # -1.0 ~ +1.0 -> -180 ~ +180 -> -3.1415 ~ +3.1415
      q1w_cos[c] += np.cos((np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD) # -1.0 ~ +1.0 -> -180 ~ +180 -> -3.1415 ~ +3.1415

      p_q1w_sin[c, i] = np.sin((np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD) # -1.0 ~ +1.0 -> -180 ~ +180 -> -3.1415 ~ +3.1415
      p_q1w_cos[c, i] = np.cos((np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD) # -1.0 ~ +1.0 -> -180 ~ +180 -> -3.1415 ~ +3.1415

      e2x[c] = test_Y[c, 0] # x
      e2y[c] = test_Y[c, 1] # y
      e2z[c] = test_Y[c, 2] # z
      q2x[c] = test_Y[c, 3] # vx
      q2y[c] = test_Y[c, 4] # vy
      q2z[c] = test_Y[c, 5] # vz
      q2w[c] = test_Y[c, 6] # w around axis-z

  e1x_mean = e1x / len(infs)
  e1y_mean = e1y / len(infs)
  e1z_mean = e1z / len(infs)
  q1x_mean = q1x / len(infs)
  q1y_mean = q1y / len(infs)
  q1z_mean = q1z / len(infs)

  # radian -> degree -> -1 ~ +1
  q1w_mean =  (np.arctan2(q1w_sin/len(infs), q1w_cos/len(infs))  * RAD2DEG) / PI_DEGREE #  -3.1415 ~ +3.1415 -> -180 ~ +180 -> -1.0 ~ +1.0
  q1w_sin_mean = q1w_sin/len(infs)
  q1w_cos_mean = q1w_cos/len(infs)
  # get var
  e1x_s = np.zeros((len(test_Y)),dtype=np.float)
  e1y_s = np.zeros((len(test_Y)),dtype=np.float)
  e1z_s = np.zeros((len(test_Y)),dtype=np.float)
  q1x_s = np.zeros((len(test_Y)),dtype=np.float)
  q1y_s = np.zeros((len(test_Y)),dtype=np.float)
  q1z_s = np.zeros((len(test_Y)),dtype=np.float)
  q1w_s = np.zeros((len(test_Y)),dtype=np.float)
  for i in range (len(infs)):
    prd = infs[i]
    for c in range(len(test_Y)):
      e1x_s[c] += np.power(prd[c, 0] - e1x_mean[c], 2)
      e1y_s[c] += np.power(prd[c, 1] - e1y_mean[c], 2)
      e1z_s[c] += np.power(prd[c, 2] - e1z_mean[c], 2)
      q1x_s[c] += np.power(prd[c, 3] - q1x_mean[c], 2)
      q1y_s[c] += np.power(prd[c, 4] - q1y_mean[c], 2)
      q1z_s[c] += np.power(prd[c, 5] - q1z_mean[c], 2)
      # variance of rotate diff around axis-z; -1 ~ +1 -> degree -> radian
      A =  (q1w_mean[c] * PI_DEGREE) * DEG2RAD
      B =  (np.clip(prd[c, 6], -1.0, 1.0) * PI_DEGREE) * DEG2RAD # -1 ~ +1 -> degree -> radian
      diff = np.abs(np.arctan2(np.sin(A - B), np.cos(A - B)))  * RAD2DEG # radian -> degree
      q1w_s[c] += diff # degree loss

  e1x_s = np.sqrt(e1x_s) / len(infs) # norm unit (take sqrt)
  e1y_s = np.sqrt(e1y_s) / len(infs) # norm unit
  e1z_s = np.sqrt(e1z_s) / len(infs) # norm unit
  q1x_s = np.sqrt(q1x_s) / len(infs) # norm unit (take sqrt)
  q1y_s = np.sqrt(q1y_s) / len(infs) # norm unit
  q1z_s = np.sqrt(q1z_s) / len(infs) # norm unit
  q1w_s = q1w_s / len(infs) # degree unit

  # get loss
  lex = np.abs(e1x_mean - e2x)
  ley = np.abs(e1y_mean - e2y)
  lez = np.abs(e1z_mean - e2z)
  lx = np.abs(q1x_mean - q2x)
  ly = np.abs(q1y_mean - q2y)
  lz = np.abs(q1z_mean - q2z)
  A =  (q1w_mean * PI_DEGREE) * DEG2RAD
  B =  (q2w * PI_DEGREE) * DEG2RAD
  lw = np.abs(np.arctan2(np.sin(A - B), np.cos(A - B)))  * RAD2DEG # rad -> degree

  error_sums = 0
  error_losses = 0
  error_losses_xyz = 0
  error_losses_vxvyvz = 0
  error_losses_w = 0
  print("a: ground truth, p: predict, l: loss, s: variance; w loss is degree unit")
  for c in range(len(test_Y)):
    # var
    error_sum = e1x_s[c] + e1y_s[c] + e1z_s[c] + q1x_s[c] + q1y_s[c] + q1z_s[c] + ((q1w_s[c]/PI_DEGREE)*2) # degree -> 0~2
    error_sums += error_sum
    # total error
    error_loss = lex[c] + ley[c] + lez[c] + lx[c] + ly[c] + lz[c]# + ((lw[c]/PI_DEGREE)*2)
    #error_loss = (lw[c]/PI_DEGREE)*2
    error_losses += error_loss

    # each error
    error_losses_xyz += (lex[c] + ley[c] + lez[c])
    error_losses_vxvyvz += (lx[c] + ly[c] + lz[c])
    error_losses_w += ((lw[c]/PI_DEGREE)*2) # loss, 0 ~ +2

    e1x_scale = ds.min_max_decode_n1_1(e1x_mean[c], s_minx, s_maxx)
    e1y_scale = ds.min_max_decode_n1_1(e1y_mean[c], s_miny, s_maxy)
    e1z_scale = ds.min_max_decode_n1_1(e1z_mean[c], s_minz, s_maxz)
    q1x_scale = ds.min_max_decode_n1_1(q1x_mean[c], s_minnx, s_maxnx)
    q1y_scale = ds.min_max_decode_n1_1(q1y_mean[c], s_minny, s_maxny)
    q1z_scale = ds.min_max_decode_n1_1(q1z_mean[c], s_minnz, s_maxnz)
    # todo: w_scale

    e2x_scale = ds.min_max_decode_n1_1(e2x[c],      s_minx, s_maxx)
    e2y_scale = ds.min_max_decode_n1_1(e2y[c],      s_miny, s_maxy)
    e2z_scale = ds.min_max_decode_n1_1(e2z[c],      s_minz, s_maxz)
    q2x_scale = ds.min_max_decode_n1_1(q2x[c],      s_minnx, s_maxnx)
    q2y_scale = ds.min_max_decode_n1_1(q2y[c],      s_minny, s_maxny)
    q2z_scale = ds.min_max_decode_n1_1(q2z[c],      s_minnz, s_maxnz)

    w_var = 1 - (q1w_sin_mean[c] ** 2 + q1w_cos_mean[c] ** 2)
    print( " |%s| %s, sum_var[%6.4f], sum_loss[%6.4f],\n\
            x[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            y[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            z[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \n\
            vx[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            vy[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            vz[a:%6.4f (%6.4f), p:%6.4f (%6.4f), l:%6.4f, s:%6.4f], \
            w[a:%6.4f (%6.4f), p:%6.4f (%6.4f), v: %6.4f, l:%6.4f, s:%6.4f]" % (
        str(c).zfill(4), test_filename[c], error_sum, error_loss,
        e2x[c], e2x_scale, e1x_mean[c], e1x_scale, lex[c], e1x_s[c],
        e2y[c], e2y_scale, e1y_mean[c], e1y_scale, ley[c], e1y_s[c],
        e2z[c], e2z_scale, e1z_mean[c], e1z_scale, lez[c], e1z_s[c],
        q2x[c], q2x_scale, q1x_mean[c], q1x_scale, lx[c], q1x_s[c],
        q2y[c], q2y_scale, q1y_mean[c], q1y_scale, ly[c], q1y_s[c],
        q2z[c], q2z_scale, q1z_mean[c], q1z_scale, lz[c], q1z_s[c],
        q2w[c], q2w[c] * PI_DEGREE, q1w_mean[c], q1w_mean[c] * PI_DEGREE, w_var, lw[c], q1w_s[c]))

  print("---------------------------------------------------CSV---------------------------------------------------")
  print( "num, filename, sum_var, sum_loss, x[a], x[a_scale], x[p], x[p_scale], x[l], x[s], y[a], y[a_scale], y[p], y[p_scale], y[l], y[s], z[a], z[a_scale], z[p], z[p_scale], z[l], z[s], vx[a],vx[a_scale], vx[p], vx[p_scale], vx[l], vx[s], vy[a], vy[a_scale], vy[p], vy[p_scale], vy[l], vy[s], vz[a], vz[a_scale], vz[p], vz[p_scale], vz[l], vz[s], w[a], w[a-deg], w[p], w[p-deg], w[v], w[l], w[s]" )
  for c in range(len(test_Y)):
    # var
    error_sum = e1x_s[c] + e1y_s[c] + e1z_s[c] + q1x_s[c] + q1y_s[c] + q1z_s[c] + ((q1w_s[c]/PI_DEGREE)*2) # degree -> 0~2
    # loss
    error_loss = lex[c] + ley[c] + lez[c] + lx[c] + ly[c] + lz[c] + ((lw[c]/PI_DEGREE)*2)

    # scale
    e1x_scale = ds.min_max_decode_n1_1(e1x_mean[c], s_minx, s_maxx)
    e1y_scale = ds.min_max_decode_n1_1(e1y_mean[c], s_miny, s_maxy)
    e1z_scale = ds.min_max_decode_n1_1(e1z_mean[c], s_minz, s_maxz)
    q1x_scale = ds.min_max_decode_n1_1(q1x_mean[c], s_minnx, s_maxnx)
    q1y_scale = ds.min_max_decode_n1_1(q1y_mean[c], s_minny, s_maxny)
    q1z_scale = ds.min_max_decode_n1_1(q1z_mean[c], s_minnz, s_maxnz)

    e2x_scale = ds.min_max_decode_n1_1(e2x[c],      s_minx, s_maxx) 
    e2y_scale = ds.min_max_decode_n1_1(e2y[c],      s_miny, s_maxy)
    e2z_scale = ds.min_max_decode_n1_1(e2z[c],      s_minz, s_maxz)
    q2x_scale = ds.min_max_decode_n1_1(q2x[c],      s_minnx, s_maxnx)
    q2y_scale = ds.min_max_decode_n1_1(q2y[c],      s_minny, s_maxny)
    q2z_scale = ds.min_max_decode_n1_1(q2z[c],      s_minnz, s_maxnz)

    w_var = 1 - (q1w_sin_mean[c] ** 2 + q1w_cos_mean[c] ** 2)
    print( "%s, %s, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f, %6.4f" % (
        str(c).zfill(4), test_filename[c], error_sum, error_loss, 
        e2x[c], e2x_scale, e1x_mean[c], e1x_scale, lex[c], e1x_s[c],
        e2y[c], e2y_scale, e1y_mean[c], e1y_scale, ley[c], e1y_s[c],
        e2z[c], e2z_scale, e1z_mean[c], e1z_scale, lez[c], e1z_s[c],
        q2x[c], q2x_scale, q1x_mean[c], q1x_scale, lx[c] , q1x_s[c],
        q2y[c], q2y_scale, q1y_mean[c], q1y_scale, ly[c] , q1y_s[c],
        q2z[c], q2z_scale, q1z_mean[c], q1z_scale, lz[c] , q1z_s[c],
        q2w[c], q2w[c] * PI_DEGREE, q1w_mean[c], q1w_mean[c] * PI_DEGREE, w_var, lw[c], q1w_s[c]))

  print("errors (var): %f" % (error_sums))
  print("errors (loss): %f, xyz: %f, vxvyvz: %f, w: %f" % (error_losses, error_losses_xyz, error_losses_vxvyvz, error_losses_w))
  return error_losses

class Save_model:
  old_error = 9999999
  now_error = -1
  dataset = []
  test_x = []
  test_y = []
  test_filename = []

  def load_test_data(self, _dataset, _x, _y, _c):
    self.dataset = _dataset
    self.test_x = _x
    self.test_y = _y
    self.test_filename = _c
    return 0

  def save_model(self, _model, _filename,  _error):
    self.now_error = _error
    if self.now_error < self.old_error:
      print("[Save evenet] Update " + str(self.old_error) + " -> " + str(self.now_error))
      self.old_error = self.now_error
      _model.save_weights(_filename)
    return 0
   
   



