//
// Created by aa on 17.05.2025.
//

#ifndef FILTERPROJECT_TEST_H
#define FILTERPROJECT_TEST_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat sharpeningOpenCv(Mat source, float laplacianBoostfactor);
Mat embossOpenCv(Mat source, int bias);
Mat fisheyeOpenCv(Mat image);
Mat tiltShiftOpenCv(Mat source, float sigma, int low, int high);

#endif //FILTERPROJECT_TEST_H
