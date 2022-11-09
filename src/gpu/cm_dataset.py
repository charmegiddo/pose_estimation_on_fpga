# import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '/workspace/91_common/'))
# import cm_dataset

import cv2
import numpy as np
import random
from keras.applications import imagenet_utils
import os

class Dataset:
    def __init__(self, classes=3, input_shape = (60, 60, 3), train_file='train.txt', test_file='test.txt', dataPath=''):
        self.train_file = train_file
        self.test_file = test_file
        self.data_shape = input_shape[0] * input_shape[1]
        self.classes = classes
        self.img_size_x = input_shape[0]
        self.img_size_y = input_shape[1]
        self.img_channel = input_shape[2]
        self.DataPath = dataPath
        self.setFileList = []


    def preprocess_inputs(self, X):
    ### @ https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
        """Preprocesses a tensor encoding a batch of images.
        # Arguments
            x: input Numpy tensor, 4D.
            data_format: data format of the image tensor.
            mode: One of "caffe", "tf".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
        # Returns
            Preprocessed tensor.
        """
        return imagenet_utils.preprocess_input(X)

    def reshape_labels(self, y):
        return np.reshape(y, (len(y), self.data_shape, self.classes))

    def normalized(self, rgb):
        #return rgb/255.0
        norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

        b=rgb[:,:,0]
        g=rgb[:,:,1]
        r=rgb[:,:,2]

        norm[:,:,0]=cv2.equalizeHist(b)
        norm[:,:,1]=cv2.equalizeHist(g)
        norm[:,:,2]=cv2.equalizeHist(r)

        return norm

    def one_hot_it(self, labels):
        x = np.zeros([360,480, self.classes])
        for i in range(360):
            for j in range(480):
                x[i,j,labels[i][j]] = 1
        return x

    def min_max_normalization(x):
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / ( x_max - x_min)
        return x_norm

    ## normilization
    def min_max_encode(self, x, source_min, source_max):
        #return (x - source_min) / ( source_max - source_min) # 0 ~ +1
        return (((x - source_min) / ( source_max - source_min) ) * 2) - 1.0 # -1 ~ +1

    def min_max_decode(self, x, source_min, source_max):
        return ((x * source_max) - (x * source_min) + source_max + source_min) * 0.5 # -1 ~ +1

    def min_max_encode_0_1(self, x, source_min, source_max):
        return ((x - source_min) / ( source_max - source_min) )

    def min_max_decode_0_1(self, x, source_min, source_max):
        return (x * source_max) - (x * source_min) + source_min 

    def min_max_encode_n1_1(self, x, source_min, source_max):
        return (((x - source_min) / ( source_max - source_min) ) * 2) - 1.0 # -1 ~ +1

    def min_max_decode_n1_1(self, x, source_min, source_max):
        return ((x * source_max) - (x * source_min) + source_max + source_min) * 0.5 # -1 ~ +1


    def get_min_max(self, filename="", read_culmn=0):
      # ファイルを開く
      tmp_line = []
      f = open(self.DataPath + filename, 'r')
      for line in f:
        tmp_line.append(line.rstrip())

      data = []
      for i in range(len(tmp_line)):
        parse = tmp_line[i].split(',')
        data.append(float(parse[read_culmn])) 
     
      data = np.asarray(data)
      return data.min(), data.max()

    def load_debris_rocket(self, _num = 10, _norm_range=[], filename="", randomly = True):
      images = np.zeros((_num, self.img_size_x, self.img_size_y, self.img_channel),dtype=np.float)
      labels = np.zeros((_num, self.classes), dtype=np.float)
      self.setFileList = []
      s_minx, s_maxx = _norm_range[0]
      s_miny, s_maxy = _norm_range[1]
      s_minz, s_maxz = _norm_range[2]
      s_minnx, s_maxnx = _norm_range[3]
      s_minny, s_maxny = _norm_range[4]
      s_minnz, s_maxnz = _norm_range[5]
      s_minrw, s_maxrw = _norm_range[6]

      # ファイルを開く
      tmp_line = []
      f = open(self.DataPath + filename, 'r')
      for line in f:
        tmp_line.append(line.rstrip())
      
      if randomly == True:
        tmp_line = random.sample(tmp_line, _num)

      for i in range(_num):
        parse = tmp_line[i].split(',')
        #print(self.DataPath+ parse[0])
        # read data
        img = cv2.imread(self.DataPath + parse[0])
        self.setFileList.append(parse[0])
        img = cv2.resize(img, (self.img_size_x, self.img_size_y))
        img = img.transpose(1, 0, 2) # y, x, c -> x, y, c
        img = np.reshape(img, (self.img_size_x, self.img_size_y, self.img_channel))
        # convert to 1 from 0 
        images[i, :, :, :] = (img / 255.0) 
        labels[i, 0] = self.min_max_encode_n1_1(float(parse[1]),   s_minx,  s_maxx) # x
        labels[i, 1] = self.min_max_encode_n1_1(float(parse[2]),   s_miny,  s_maxy) # y
        labels[i, 2] = self.min_max_encode_n1_1(float(parse[3]),   s_minz,  s_maxz) # z
        labels[i, 3] = self.min_max_encode_n1_1(float(parse[11]),  s_minnx, s_maxnx) # nx
        labels[i, 4] = self.min_max_encode_n1_1(float(parse[12]),  s_minny, s_maxny) # ny
        labels[i, 5] = self.min_max_encode_n1_1(float(parse[13]),  s_minnz, s_maxnz) # nz
        labels[i, 6] = self.min_max_encode_n1_1(float(parse[14]),  s_minrw, s_maxrw) # rw 
      f.close()
      return (images, labels)


