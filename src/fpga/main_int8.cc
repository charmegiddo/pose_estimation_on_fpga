/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

GraphInfo shapes;

//const string baseImagePath = "./test/";
//const string wordsPath = "./";
string baseImagePath, wordsPath;  // they will get their values via argv[]

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images - the vector of image name
 *
 * @return none
 */
void ListImages(string const& path, vector<string>& images) {
  images.clear();
  struct dirent* entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode)) {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN) {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png")) {
        images.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief load kinds from file to a vector
 *
 * @param path - path of the kinds file
 * @param kinds - the vector of kinds string
 *
 * @return none
 */
void LoadWords(string const& path, vector<string>& kinds) {
  kinds.clear();
  ifstream fkinds(path);
  if (fkinds.fail()) {
    fprintf(stderr, "Error : Open %s failed.\n", path.c_str());
    exit(1);
  }
  string kind;
  while (getline(fkinds, kind)) {
    kinds.push_back(kind);
  }

  fkinds.close();
}

/**
 * @brief calculate softmax
 *
 * @param data - pointer to input buffer
 * @param size - size of input buffer
 * @param result - calculation result
 *
 * @return none
 */
void CPUCalcSoftmax(const int8_t* data, size_t size, float* result, float scale) {
  assert(data && result);
  double sum = 0.0f;

  for (size_t i = 0; i < size; i++) {
    result[i] = exp((float)data[i] * scale);
    sum += result[i];
  }
  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

/**
 * @brief Get top k results according to its probability
 *
 * @param d - pointer to input data
 * @param size - size of input data
 * @param k - calculation result
 * @param vkinds - vector of kinds
 *
 * @return none
 */
void TopK(const float* d, int size, int k, vector<string>& vkinds) {
  assert(d && size > 0 && k > 0);
  priority_queue<pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(pair<float, int>(d[i], i));
  }

  for (auto i = 0; i < k; ++i) {
    pair<float, int> ki = q.top();
    printf("top[%d] prob = %-8f  name = %s\n", i, d[ki.second],
           vkinds[ki.second].c_str());
    q.pop();
  }
}


float min_max_decode_n1_1(float _x, float _min, float _max){
	return  ( (_x * _max) - (_x * _min) + _max + _min ) * 0.5;
}


/**
 * @brief Run DPU Task for CNN
 *
 * @return none
 */
void run_CNN(vart::Runner* runner) {
  /////////////////////////////////////////////////////////////////////////////////////////////
  // TIMERS CALIBRATION

  int num_of_trials = 200;
  std::chrono::duration<double, std::micro> avg_calibr_highres(0);
  for (int i =0; i<num_of_trials; i++)
  {
      auto t1 = std::chrono::high_resolution_clock::now();
      auto t2 = std::chrono::high_resolution_clock::now();
      // floating-point duration: no duration_cast needed
      std::chrono::duration<double, std::micro> fp_us = t2 - t1;
      avg_calibr_highres  += fp_us;
      //if (i%10 ==0) cout << "[Timers calibration  ] " << fp_us.count() << "us" << endl;
    }
  
  avg_calibr_highres  /= num_of_trials;
  cout << "[average calibration high resolution clock] " << avg_calibr_highres.count() << "us"  << endl;
  cout << "\n" << endl;
  /////////////////////////////////////////////////////////////////////////////////////////////

  vector<string> kinds, images, results, log1, log2, log3;

  /* Load all image names.*/
  ListImages(baseImagePath, images);
  if (images.size() == 0) {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  /* Load all kinds words.*/
  //LoadWords(wordsPath + "labels.txt", kinds);
  LoadWords(wordsPath, kinds);
  if (kinds.size() == 0) {
    cerr << "\nError: No words exist in file words.txt." << endl;
    return;
  }
  //float mean[3] = {104, 107, 123};

  /* get in/out tensors and dims*/
  auto outputTensors = runner->get_output_tensors();
  auto inputTensors = runner->get_input_tensors();
  auto out_dims = outputTensors[0]->get_shape(); //_dims();
  auto in_dims = inputTensors[0]->get_shape(); //dims();

  auto input_scale = get_input_scale(inputTensors[0]);
  auto output_scale = get_output_scale(outputTensors[0]);

  /*get shape info*/
  int outSize = shapes.outTensorList[0].size;
  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int batchSize = in_dims[0];

  //for debug
  //cout << "OUT  dims " << out_dims  << endl;
  cout << "OUT  size " << outSize   << endl;
  //cout << "IN   dims " << in_dims   << endl;
  cout << "IN   size " << inSize    << endl;
  cout << "IN Height " << inHeight  << endl;
  cout << "IN Width  " << inWidth   << endl;
  cout << "batchSize " << batchSize << endl;

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;

  vector<Mat> imageList;
  int8_t* imageInputs = new int8_t[inSize * batchSize];

  float* softmax = new float[outSize];
  int8_t* FCResult = new int8_t[batchSize * outSize];
  std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  /*run with batch*/
  for (unsigned int n = 0; n < images.size(); n += batchSize) {


    auto t1 = chrono::high_resolution_clock::now();
    unsigned int runSize =
        (images.size() < (n + batchSize)) ? (images.size() - n) : batchSize;
    in_dims[0] = runSize;
    out_dims[0] = batchSize;
    for (unsigned int i = 0; i < runSize; i++) {
      Mat image = imread(baseImagePath + images[n + i]);
      /*image pre-process*/
      Mat image2 = cv::Mat(inHeight, inWidth, CV_8SC3);
      resize(image, image2, Size(inWidth, inHeight), 0, 0, INTER_LINEAR);

      for (int h = 0; h < inHeight; h++) {
        for (int w = 0; w < inWidth; w++) {
          for (int c = 0; c < 3; c++) {
          //imageInputs[i*inSize+h*inWidth*3+w*3 +  c] = (int8_t)( (image2.at<Vec3b>(h, w)[c]/255.0f-0.5f)*2*input_scale ); //in BGR mode
            imageInputs[i*inSize+h*inWidth*3+w*3 +  c] = (int8_t)( ((image2.at<Vec3b>(h, w)[c]/255.0f*2.0f)-1)*input_scale); //in BGR mode
	  //imageInputs[i*inSize+h*inWidth*3+w*3 +2-c] = (int8_t)( (image2.at<Vec3b>(h, w)[c]/255.0f-0.5f)*2*input_scale ); //in RGB mode
          }
	 // cout << "w:" << w << ", h:" << h  << ", " << +(image2.at<Vec3b>(h, w)[0]) << "," << +(image2.at<Vec3b>(h, w)[1]) << "," << +(image2.at<Vec3b>(h, w)[2]) << endl;
        }
      }
      imageList.push_back(image);
    }
    auto t2 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> diff1 = (t2 - t1) - avg_calibr_highres;
    double diff1_time = (double) diff1.count();
    cout << "[Load image tot Time ] " << diff1_time  << "ms" << endl;
    log1.push_back(std::to_string(diff1_time));


    auto start_time = chrono::high_resolution_clock::now();

    /* in/out tensor refactory for batch inout/output */
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        imageInputs, batchTensors.back().get()));
    batchTensors.push_back(std::shared_ptr<xir::Tensor>(
        xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::XINT, 8u})));
    outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
        FCResult, batchTensors.back().get()));

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> dpu_time = (end_time - start_time) - avg_calibr_highres;
    double dpu_tot_time = (double) dpu_time.count();
    cout << "[DPU tot Time ] " << dpu_tot_time  << "us" << endl;
    log2.push_back(std::to_string(dpu_tot_time));

    /*tensor buffer input/output */
    inputsPtr.clear();
    outputsPtr.clear();
    inputsPtr.push_back(inputs[0].get());
    outputsPtr.push_back(outputs[0].get());

    /*run*/
    auto t3 = chrono::high_resolution_clock::now();
    auto job_id = runner->execute_async(inputsPtr, outputsPtr);
    runner->wait(job_id.first, -1);
    for (unsigned int i = 0; i < runSize; i++) {
      //cout << "\nImage : " << images[n + i] << endl;
      float x,y,z,vx,vy,vz;
      float dx,dy,dz,dvx,dvy,dvz;

      // each result
      const int8_t* data = &FCResult[i * outSize];
      for (size_t j = 0; j < outSize; j++) {
        //cout << (float)data[j] << ", " << ((float)data[j] * output_scale) << endl;
	if(j == 0){
		x = ((float)data[j] * output_scale) ;
		dx = min_max_decode_n1_1(x, -14 , 14);
	}
	if(j == 1){
	       	y = ((float)data[j] * output_scale) ; 
		dy = min_max_decode_n1_1(y, -10.4 , 10.4);
	}
	if(j == 2){
	       	z = ((float)data[j] * output_scale) ;
		dz = min_max_decode_n1_1(z, 30 , 70);
	}
	if(j == 3){ vx = ((float)data[j] * output_scale); }
	if(j == 4){ vy = ((float)data[j] * output_scale); }
	if(j == 5){ vz = ((float)data[j] * output_scale); }
      }
      float length = sqrt(((vx * vx) + (vy * vy) + (vz * vz)));
      dvx = vx / length;
      dvy = vy / length;
      dvz = vz / length;
      //cout << images[n+i] << "," << x << "," << y << "," << z << "," << vx << "," << vy << "," << vz << endl;
      //cout << images[n+i] << "," << dx << "," << dy << "," << dz << "," << dvx << "," << dvy << "," << dvz << endl;
      std::string tmp = images[n+i] + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(vx) + ", " + std::to_string(vy) + ", " + std::to_string(vz);
      results.push_back(tmp);

    }
    auto t4 = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> diff2 = (t4 - t3) - avg_calibr_highres;
    double diff2_time = (double) diff2.count();
    cout << "[Load image tot Time ] " << diff2_time  << "ms" << endl;
    log3.push_back(std::to_string(diff2_time));
    imageList.clear();
    inputs.clear();
    outputs.clear();
  }
  delete[] FCResult;
  delete[] imageInputs;
  delete[] softmax;

  for (int i = 0; i < results.size(); i++){
	  cout << results[i] << endl;
  }
  cout << endl;
  cout << "log1" << endl;
  for (int i = 0; i < log1.size(); i++){
	  cout << log1[i] << ",";
  }
  cout << endl;
  cout << "log2" << endl;
  for (int i = 0; i < log2.size(); i++){
	  cout << log2[i] << ",";
  }
  cout << endl;
  cout << "log3" << endl;
  for (int i = 0; i < log3.size(); i++){
	  cout << log3[i] << ",";
  }

}

/**
 * @brief Entry for running CNN
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy CNN on DPU platform.
 *
 */
int main(int argc, char* argv[])
{
  // Check args
  if (argc != 4) {
    cout << "Usage: run_cnn elf_pathName test_images_pathname, labels_filename" << endl;
    return -1;
  }

  baseImagePath = std::string(argv[2]); //path name of the folder with test images
  wordsPath     = std::string(argv[3]); //filename of the labels

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
  /*create runner*/
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
  /*get in/out tensor*/
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  /*get in/out tensor shape*/
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  /*run with batch*/
  run_CNN(runner.get());
  return 0;
}
