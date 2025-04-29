//
// Created by aa on 24.03.2025.
//

#ifndef FILTERPROJECT_EFFECTS_H
#define FILTERPROJECT_EFFECTS_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat bgr_2_grayscale(Mat source);

Mat mySharpening(Mat source, float laplacianBoostfactor);

Mat sharpeningOpenCv(Mat source, float laplacianBoostfactor);

Mat myEmboss(Mat source, int bias);

Mat myFisheye(Mat source, float k);

Mat fisheyeCircular(Mat source, float k);

Mat createGaussianKernel(float sigma);

Mat tiltShift(Mat source, float sigma, int low, int high);

Mat maskEffect(Mat mask, Mat source, Mat effect);


#endif //FILTERPROJECT_EFFECTS_H
