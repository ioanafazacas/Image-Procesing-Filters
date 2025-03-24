//
// Created by aa on 24.03.2025.
//

#ifndef FILTERPROJECT_EFFECTS_H
#define FILTERPROJECT_EFFECTS_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

Mat bgr_2_grayscale(Mat source);

Mat sharpening(Mat source, float laplacianBoostfactor);

Mat sharpeningOpenCv(Mat source, float laplacianBoostfactor);

Mat myEmboss(Mat source, int bias);


#endif //FILTERPROJECT_EFFECTS_H
