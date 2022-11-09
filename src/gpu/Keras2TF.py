#!/usr/bin/env python
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

import os, sys, shutil

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import adda_model as am
import adda_utils as au
import tensorflow.keras
import argparse #DB
ap = argparse.ArgumentParser()
ap.add_argument("-n",  "--hdf", default="",          help="input CNN")
args = vars(ap.parse_args())
hdf_dir = args["hdf"]
print("file read: ", hdf_dir)

##############################################
# Set up directories
##############################################
# set learning phase for no training: This line must be executed before loading Keras model
K.set_learning_phase(0)

# load model
#extractor_feature_model = am.build_extractor_feature_model()
#pose_estimation_model = am.build_pose_estimation_model(_input_shape=extractor_feature_model.output.shape[1:])
#print(extractor_feature_model.summary())
#print(extractor_feature_model.layers)
#print(pose_estimation_model.summary())
#print(pose_estimation_model.layers)
#prediction_model = am.concat_model(extractor_feature_model, pose_estimation_model)
#opt = tf.keras.optimizers.Adam(lr=1e-4) 
#prediction_model.compile(loss=au.dloss_function, optimizer=opt)
prediction_model = am.build_model()
prediction_model.load_weights(hdf_dir)

# load weights & architecture into new model
#model = load_model(hdf_dir, compile=False)
model = prediction_model

#print the CNN structure
model.summary()

# make list of output node names
output_names=[out.op.name for out in model.outputs]


# set up tensorflow saver object
saver = tf.compat.v1.train.Saver()

# fetch the tensorflow session using the Keras backend
sess = tf.compat.v1.keras.backend.get_session()

## get the tensorflow session graph
#graph_def = sess.graph.as_graph_def()


# Check the input and output name
print ("\n TF input node name:")
print(model.inputs)
print ("\n TF output node name:")
print(model.outputs)
print("\n All Node Name")
names =[l.output for l in model.layers]
print(names)

# write out tensorflow checkpoint & inference graph (from MH's "MNIST classification with TensorFlow and Xilinx DNNDK")
save_path = saver.save(sess, "/workspace/build/tf_chkpts/debris_data/debris_network/float_model.ckpt")


print ("\nFINISHED CREATING TF FILES\n")
