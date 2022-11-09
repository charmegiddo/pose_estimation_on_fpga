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

import cv2
import os
import numpy as np
import cm_dataset

# image load settings
width = int(1636 * 0.5)
height = int(1220 * 0.5)
channels = 3
input_shape = (height, width, channels)
classes = 7 # nx, ny, nz, z-nw
norm_range = [(-14, 14),(-10.4, 10.4),(30.0, 70.0),(-1, 1),(-1, 1),(-1, 1),(-180, 180)]
ds = cm_dataset.Dataset(classes=7, input_shape=input_shape, dataPath='/workspace/dataset/test/')
# settings
calib_batch_size = 30

def calib_input(iter):
 # images = []
  test_X, test_Y = ds.load_debris_rocket_hw(_num=calib_batch_size, _norm_range=norm_range, filename="test.log", randomly=True)   
  return {"input_1": test_X}


#######################################################

def main():
  calib_input(0)


if __name__ == "__main__":
    main()
