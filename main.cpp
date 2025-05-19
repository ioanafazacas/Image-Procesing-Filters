#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/effects.h"
#include "src/test.h"

using namespace std;
using namespace cv;

int main() {

    Mat image = imread("D:\\An III sem II\\PI\\FilterProject\\images\\flowers_24bits.bmp",IMREAD_COLOR);
    Mat imageCity = imread("D:\\An III sem II\\PI\\FilterProject\\images\\city-lights.bmp",IMREAD_COLOR);
    Mat imageCity2 = imread("D:\\An III sem II\\PI\\FilterProject\\images\\city2.bmp",IMREAD_COLOR);


    imshow("original",image);
/*
    Mat sharpening = mySharpening(image,1);
    Mat sharpening_opencv = sharpeningOpenCv(image,1);
    imshow("sharpening",sharpening);
    imshow("openCV",sharpening_opencv);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\sharpening_1.bmp",sharpening);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\sharpening_1_opencv.bmp",sharpening_opencv);

    Mat emboss = myEmboss(image,0);
    Mat emboss_opencv = embossOpenCv(image,0);
    imshow("emboss",myEmboss(image,0));
    imshow("emboss OpenCv",embossOpenCv(image,0));
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\emboss_0.bmp",emboss);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\emboss_0_opencv.bmp",emboss_opencv);

    Mat sepiaImg = sepia(image);
    Mat negativImg = negativ(image);
    imshow("sepia",sepiaImg);
    imshow("negativ",negativImg);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\sepia1.bmp",sepiaImg);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\negativ1.bmp",negativImg);


    //imshow("fisheye2 1",myFisheye2(image,1));
    imshow("fisheye2 1.5",myFisheye2(image,1.5));
    imshow("fisheye2 2",myFisheye2(image,2));
    imshow("fisheye2 3",myFisheye2(image,3));
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\fisheye_15.bmp",myFisheye2(image,1.5));
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\fisheye_2.bmp",myFisheye2(image,2));
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\fisheye_3.bmp",myFisheye2(image,3));
    //imshow("fisheye 1",myFisheye(image,1));
    //imshow("fisheye 1.5",myFisheye(image,1.5));
    //imshow("fisheye 3",myFisheye(image,3));
    //imshow("fisheye2",myFisheye2(imageCity,3));
    //imshow("fisheye",myFisheye(imageCity,3));

    imshow("tilt shift", tiltShift(imageCity,1.5, 400,500));
    imshow("tilt shift opencv", tiltShiftOpenCv(imageCity,1.5, 400,500));
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\tiltShift.bmp",tiltShift(imageCity,1.5, 400,500));
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\tiltShift_opencv.bmp",tiltShiftOpenCv(imageCity,1.5, 400,500));

    //la toate functiile sa aplic si implementarea din openCV , iar la partea de testare sa le compar cele doua penru noise
    //pentru comparare folosesc PSNR, SSIM, and LPIPS
    //imshow("original city", imageCity2);
    //imshow("tilt shift",tiltShift(imageCity,1.2, 30,100));

    Mat myBokeh = bokehCerc(imageCity);
    Mat onlineBokeh = bokehEffect(imageCity,10);
    imshow("bokeh 2",onlineBokeh);
    imshow("bokeh",myBokeh);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\bokeh.bmp",myBokeh);
    imwrite("D:\\An III sem II\\PI\\FilterProject\\results\\bokeh_online.bmp",onlineBokeh);
    */

    waitKey(0);
    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.