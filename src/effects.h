//
// Created by aa on 24.03.2025.
//

#ifndef FILTERPROJECT_EFFECTS_H
#define FILTERPROJECT_EFFECTS_H
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

typedef struct {
    Mat labels;
    int no_labels;
    vector<Vec3b> colors;
}labels;

const int n8_di[8] = {0,-1,-1, -1, 0, 1, 1, 1};
const int n8_dj[8] = {1, 1, 0, -1, -1,-1, 0, 1};

const int np_di[4] = { 0,-1,-1, -1};
const int np_dj[4] = { -1,-1, 0, 1};


Mat mySharpening(Mat source, float laplacianBoostfactor);

Mat myEmboss(Mat source, int bias);

Mat sepia(Mat source);

Mat negativ(Mat source);

Mat myGaussianBlur(Mat source, float sigma);

Mat myFisheye(Mat source, float k);

Mat myFisheye2(Mat source, float k);

Mat tiltShift(Mat source, float sigma, int low, int high);

Mat maskEffect(Mat mask, Mat source, Mat effect);

Mat bokehEffect(const Mat& input, int radius);

labels BFS_labeling(Mat source);

Mat bokehCerc(Mat source);


#endif //FILTERPROJECT_EFFECTS_H