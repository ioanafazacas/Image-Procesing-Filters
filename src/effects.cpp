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

Mat mySharpening(Mat source, float laplacianBoostfactor){
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

Mat myFisheye(Mat source, float k){
    Point origin = Point(source.cols/2,source.rows/2);
    Mat result = Mat::zeros(source.size(), source.type());
    int newX, newY;
    float xn, yn, x2, y2;
    double r;
    /*
    for(int i=0; i<source.rows; i++){
        for(int j=0; j<source.cols; j++){
            x=j-origin.x;
            y=i-origin.y;
            r=sqrt(x*x+y*y);
            newX= x*(1/(1+k*r*r))+origin.x;
            newY= y*(1/(1+k*r*r))+origin.y;
            if(IsInside(result,newY,newX)){
                result.at<Vec3b>(newY,newX)= source.at<Vec3b>(i,j);
            }
        }
    }*/
    float cx=source.cols/2.0f; //center
    float cy=source.rows/2.0f;
    for(int y=0; y<source.rows; y++){
        for(int x=0; x<source.cols; x++){
            xn=(float)(x-cx)/cx;
            yn=(float)(y-cy)/cy;
            r=sqrt(xn*xn+yn*yn);
            // Aplica distorsiunea fisheye
            float rDistorted = (r == 0.0f) ? 0.0f : atan(r * k) / (r * k);

            float xMapped = cx + xn * rDistorted * cx;
            float yMapped = cy + yn * rDistorted * cy;

            // Interpolare nearest-neighbor (poți schimba cu biliniară pentru calitate mai bună)
            int xSrc = round(xMapped);
            int ySrc = round(yMapped);

            if (IsInside(result,ySrc,xSrc)) {
                result.at<Vec3b>(y, x) = source.at<Vec3b>(ySrc, xSrc);
            }
        }
    }

    return result;
}

Mat fisheyeCircular(Mat source, float k) {
    int w = source.cols;
    int h = source.rows;
    Point2f center(w / 2.0f, h / 2.0f);
    float maxRadius = std::min(w, h) / 2.0f;

    Mat result = Mat::zeros(source.size(), source.type());

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = x - center.x;
            float dy = y - center.y;
            float r = sqrt(dx * dx + dy * dy);

            //if (r < maxRadius) {
                float theta = atan2(dy, dx);
                float r_norm = r / maxRadius;
                float r_distorted = pow(r_norm, k) * maxRadius;

                float srcX = center.x + r_distorted * cos(theta);
                float srcY = center.y + r_distorted * sin(theta);

                if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h) {
                    result.at<Vec3b>(y, x) = source.at<Vec3b>((int)srcY, (int)srcX);
                }
            //}
        }
    }

    return result;
}

Mat createGaussianKernel(float sigma){
    Mat kernel = Mat(5,5,CV_64FC1);
    double sum=0;
    for(int x=-2; x<=2; x++){
        for(int y=-2; y<=2; y++){
            kernel.at<double>(x+2,y+2)=exp(-(float)(x*x+y*y)/(2*sigma*sigma))/(2*CV_PI*sigma*sigma);
            sum = sum+kernel.at<double>(x+2,y+2);
        }
    }
    for(int x=0; x<5; x++){
        for(int y=0; y<5; y++){
            kernel.at<double>(x,y)= kernel.at<double>(x,y)/sum;
            printf("%f ",kernel.at<double>(x,y));
        }
        printf("\n");
    }
    printf("Gaussian kernel sum %f",sum);
    return kernel;
}

Mat tiltShift(Mat source, float sigma, int low, int high){
    Mat effect = Mat::zeros(source.size(), source.type());
    Mat kernel = createGaussianKernel(sigma);
    for(int x=0; x<source.rows; x++){
        for(int y=0; y<source.cols; y++){
            Vec3d filterPixel= Vec3d (0,0,0);
            for(int i=-2; i<=2; i++){
                for(int j=-2; j<=2; j++){
                    if(IsInside(source,x+i,y+j)){
                        double k= kernel.at<double>(i+2,j+2);
                        Vec3b pixel = source.at<Vec3b>(x+i,y+j);
                        filterPixel[0] = filterPixel[0] + pixel[0] * k;
                        filterPixel[1] = filterPixel[1] + pixel[1] * k;
                        filterPixel[2] = filterPixel[2] + pixel[2] * k;
                    }
                }
            }
            effect.at<Vec3b>(x, y)[0] = cv::saturate_cast<uchar>(filterPixel[0]);
            effect.at<Vec3b>(x, y)[1] = cv::saturate_cast<uchar>(filterPixel[1]);
            effect.at<Vec3b>(x, y)[2] = cv::saturate_cast<uchar>(filterPixel[2]);
        }
    }

    Mat mask = Mat(source.rows, source.cols, CV_8UC1);
    for(int i=0; i<source.rows; i++){
        for(int j=0; j<source.cols; j++){
            if(i>low && i<high)
                mask.at<uchar>(i,j)=0;
            else mask.at<uchar>(i,j)=1;
        }
    }

    Mat result = maskEffect(mask,source,effect);
    return result;
}

Mat maskEffect(Mat mask, Mat source, Mat effect){
    Mat result = source.clone();
    for(int i=0; i<source.rows; i++){
        for(int j=0; j<source.cols; j++){
            result.at<Vec3b>(i,j)[0]=mask.at<uchar>(i,j) * effect.at<Vec3b>(i,j)[0]+ (1-mask.at<uchar>(i,j)) * source.at<Vec3b>(i,j)[0];
            result.at<Vec3b>(i,j)[1]=mask.at<uchar>(i,j) * effect.at<Vec3b>(i,j)[1]+ (1-mask.at<uchar>(i,j)) * source.at<Vec3b>(i,j)[1];
            result.at<Vec3b>(i,j)[2]=mask.at<uchar>(i,j) * effect.at<Vec3b>(i,j)[2]+ (1-mask.at<uchar>(i,j)) * source.at<Vec3b>(i,j)[2];
        }
    }
    return result;
}