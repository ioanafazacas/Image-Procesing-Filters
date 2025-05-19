//
// Created by aa on 17.05.2025.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include "test.h"
#include <cmath>

using namespace std;
using namespace cv;

Mat sharpeningOpenCv(Mat source, float laplacianBoostfactor){
    Mat result = source.clone();
    Mat kernel = (Mat_<double>(3,3)<<0, -1, 0, -1, 5*laplacianBoostfactor, -1, 0, -1, 0);
    filter2D(source.clone(),result,source.depth(),kernel,Point(-1,-1));
    return result ;
}

Mat embossOpenCv(Mat source, int bias){
    Mat result = Mat::zeros(source.size(), source.type());
    Mat kernel = (Mat_<float>(3,3)<< -2, -1, 0,
            -1, 1, 1,
            0, 1, 2);
    filter2D(source.clone(),result,source.depth(),kernel);
    return result ;
}

Mat fisheyeOpenCv(Mat image) {
    int width = image.cols;
    int height = image.rows;
    Mat result = Mat::zeros(image.size(), image.type());

    Mat map_x = Mat(image.size(), CV_32F);
    Mat map_y = Mat(image.size(), CV_32F);

    float cx = width / 2.0f;
    float cy = height / 2.0f;
    float radius = min(cx, cy);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = (x - cx) / radius;
            float dy = (y - cy) / radius;
            float r = sqrt(dx * dx + dy * dy);
            float theta = atan(r);
            float scale = (r == 0.0) ? 1.0 : theta / r;

            float src_x = cx + dx * scale * radius;
            float src_y = cy + dy * scale * radius;

            map_x.at<float>(y, x) = src_x;
            map_y.at<float>(y, x) = src_y;
        }
    }

    remap(image, result, map_x, map_y, cv::INTER_LINEAR);
    return result;
}

Mat tiltShiftOpenCv(Mat source, float sigma, int low, int high){
    Mat blurred,result;
    result = source.clone();
    int w= round(sigma * 6);
    if(w%2==0)w=w+1;
    GaussianBlur(source, blurred, Size(w, w), sigma);

    Mat mask = Mat::ones(source.size(), CV_8UC1);
    mask.rowRange(low, high).setTo(0); // zona clară

    for(int i=0; i<source.rows; i++){
        for(int j=0; j<source.cols; j++){
            result.at<Vec3b>(i,j)[0]=mask.at<uchar>(i,j) * blurred.at<Vec3b>(i,j)[0]+ (1-mask.at<uchar>(i,j)) * source.at<Vec3b>(i,j)[0];
            result.at<Vec3b>(i,j)[1]=mask.at<uchar>(i,j) * blurred.at<Vec3b>(i,j)[1]+ (1-mask.at<uchar>(i,j)) * source.at<Vec3b>(i,j)[1];
            result.at<Vec3b>(i,j)[2]=mask.at<uchar>(i,j) * blurred.at<Vec3b>(i,j)[2]+ (1-mask.at<uchar>(i,j)) * source.at<Vec3b>(i,j)[2];
        }
    }

    return result;
}

