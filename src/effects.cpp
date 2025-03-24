#include <iostream>
#include <opencv2/opencv.hpp>
#include "effects.h"

using namespace std;
using namespace cv;

Mat bgr_2_grayscale(Mat source){
    /*
    * Create a function that will convert a color RGB image (CV_8UC3 type) to a grayscale image (CV_8UC1),
    * and return the result image
    * Inputs: source: the source matrix(BGR)
    * Variables:
    *  rows: number of rows of the source matrix
    *  cols: number of cols of the source matrix
    *  grayscale_image: The grayscale image that you will obtain and return
    */
    int rows, cols;
    Mat grayscale_image;

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    grayscale_image = Mat(source.rows,source.cols,CV_8UC1);
    for(int i=0; i<source.rows; i++){
        for(int j=0; j<source.cols; j++){
            grayscale_image.at<unsigned char>(i,j)=(source.at<Vec3b>(i,j)[0]+source.at<Vec3b>(i,j)[1]+
                                                    source.at<Vec3b>(i,j)[2])/3;
        }
    }
    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return grayscale_image;

}

bool IsInside(Mat img, int i, int j){
    if(i >= 0 && i < img.rows && j >= 0 && j < img.cols)
        return true;
    return false;
}

Mat sharpeningOpenCv(Mat source, float laplacianBoostfactor){
    Mat result = source.clone();
    Mat kernel = (Mat_<double>(3,3)<<0, -1, 0, -1, 5*laplacianBoostfactor, -1, 0, -1, 0);
    filter2D(source.clone(),result,source.depth(),kernel,Point(-1,-1));
    return result ;
}

Mat sharpening(Mat source, float laplacianBoostfactor){
    //Mat result = Mat(source.rows,source.cols, CV_8UC3);
    Mat result = Mat::zeros(source.size(), source.type());
    Mat kernel = (Mat_<float>(3,3)<< 0, -1, 0,
                                                -1, 5*laplacianBoostfactor,-1,
                                                0, -1, 0);
    for(int x=0; x<source.rows; x++){
        for(int y=0; y<source.cols; y++){
            Vec3f filterPixel= Vec3f(0,0,0);
            for(int i=-1; i<=1; i++){
                for(int j=-1; j<=1; j++){
                    if(IsInside(source,x+i,y+j)){
                        float k= kernel.at<float>(i+1,j+1);
                        Vec3b pixel = source.at<Vec3b>(x+i,y+j);
                        filterPixel[0] = filterPixel[0] + pixel[0] * k;
                        filterPixel[1] = filterPixel[1] + pixel[1] * k;
                        filterPixel[2] = filterPixel[2] + pixel[2] * k;
                    }
                }
            }
            result.at<Vec3b>(x, y)[0] = cv::saturate_cast<uchar>(filterPixel[0]);
            result.at<Vec3b>(x, y)[1] = cv::saturate_cast<uchar>(filterPixel[1]);
            result.at<Vec3b>(x, y)[2] = cv::saturate_cast<uchar>(filterPixel[2]);
        }
    }
    return result ;
}

Mat myEmboss(Mat source, int bias){
    Mat result = Mat::zeros(source.size(), source.type());
    Mat kernel = (Mat_<float>(3,3)<< -2, -1, 0,
                                                -1, 1, 1,
                                                0, 1, 2);
    for(int x=0; x<source.rows; x++){
        for(int y=0; y<source.cols; y++){
            Vec3f filterPixel= Vec3f(0,0,0);
            for(int i=-1; i<=1; i++){
                for(int j=-1; j<=1; j++){
                    if(IsInside(source,x+i,y+j)){
                        float k= kernel.at<float>(i+1,j+1);
                        Vec3b pixel = source.at<Vec3b>(x+i,y+j);
                        filterPixel[0] = filterPixel[0] + pixel[0] * k;
                        filterPixel[1] = filterPixel[1] + pixel[1] * k;
                        filterPixel[2] = filterPixel[2] + pixel[2] * k;
                    }
                }
            }
            result.at<Vec3b>(x, y)[0] = cv::saturate_cast<uchar>(filterPixel[0])+bias;
            result.at<Vec3b>(x, y)[1] = cv::saturate_cast<uchar>(filterPixel[1])+bias;
            result.at<Vec3b>(x, y)[2] = cv::saturate_cast<uchar>(filterPixel[2])+bias;
        }
    }
    return result ;
}