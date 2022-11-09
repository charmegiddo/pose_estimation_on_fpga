#!/usr/binenv pytho1
# -*- coding: utf-8 -*-

'''
## © Copyright (C) 2016-2020 Xilinx, Inc
##
## Licensed under the Apache License, Version 2.0 (the "License"). You may
## not use this file except in compliance with the License. A copy of the
## License is located at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
## License for the specific language governing permissions and limitations
## under the License.
'''
# Author: Daniele Bagni, Xilinx Inc
# date 6 May 2021

##################################################################
# Evaluation of frozen/quantized graph
#################################################################

import os
import sys
import glob
import argparse
import shutil
import tensorflow as tf
import numpy as np
import cv2
import gc # memory garbage collector #DB
import numpy as np
import tensorflow.contrib.decent_q
import copy as cp
from tensorflow.python.platform import gfile
from tensorflow.keras.preprocessing.image import img_to_array

from config import cifar10_config as cfg #DB
import cm_dataset


def graph_eval(input_graph_def, input_node, output_node):
  width = int(1636 * 0.5)
  height = int(1220 * 0.5)
  channels = 3
  input_shape = (height, width, channels)
  classes = 7 # nx, ny, nz, z-nw
  norm_range = [(-14, 14),(-10.4, 10.4),(30.0, 70.0),(-1, 1),(-1, 1),(-1, 1),(-180, 180)]
  ds = cm_dataset.Dataset(classes=7, input_shape=input_shape, dataPath='/workspace/dataset/test/')
  test_X, test_Y = ds.load_debris_rocket_hw(_num=1, _norm_range=norm_range, filename="test.log", randomly=False)
  file_list = ds.get_file_list()
  # graph load
  tf.compat.v1.import_graph_def(input_graph_def,name = '')
  # Get input placeholders & tensors
  images_in = tf.compat.v1.get_default_graph().get_tensor_by_name(input_node+':0')
  labels = tf.compat.v1.placeholder(tf.float32,shape = [None, classes])
  # get output tensors
  logits = tf.compat.v1.get_default_graph().get_tensor_by_name(output_node+':0')
  # top 5 and top 1 accuracy

  # Create the Computational graph
  with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.initializers.global_variables())
    feed_dict={images_in: test_X, labels: test_Y}
    prediction = sess.run(logits, feed_dict)
  total_diff = 0
  for i in range(len(prediction)):
    diff = 0
    pred = cp.deepcopy(prediction[i])
    leng = np.sqrt((pred[3]**2) + (pred[4]**2) + (pred[5]**2))
    pred[3] /= leng
    pred[4] /= leng
    pred[5] /= leng
    diff = np.abs(pred - test_Y[i])
    total_diff += diff
    print(file_list[i])
    print(pred)
    print(test_Y[i])
    print("diff:", diff)
    print("sum diff:", np.sum(diff))
    print("--------")
  print(total_diff)
  print(np.sum(total_diff))
  print ('FINISHED!')
  return


def main(unused_argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    input_graph_def = tf.Graph().as_graph_def()
    input_graph_def.ParseFromString(tf.io.gfile.GFile(FLAGS.graph, "rb").read())
    graph_eval(input_graph_def, FLAGS.input_node, FLAGS.output_node)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str,
                        default='./freeze/frozen_graph.pb',
                        help='graph file (.pb) to be evaluated.')
    parser.add_argument('--input_node', type=str,
                        default='images_in',
                        help='input node.')
    parser.add_argument('--output_node', type=str,
                        default='dense_1/BiasAdd',
                        help='output node.')
    parser.add_argument('--class_num', type=int,
                        default=cfg.NUM_CLASSES,
                        help='number of classes.')
    parser.add_argument('--gpu', type=str,
                        default='0',
                        help='gpu device id.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
