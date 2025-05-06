#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/effects.h"

using namespace std;
using namespace cv;

int main() {

    Mat image = imread("D:\\An III sem II\\PI\\FilterProject\\images\\flowers_24bits.bmp",IMREAD_COLOR);
    Mat imageCity = imread("D:\\An III sem II\\PI\\FilterProject\\images\\city-lights.bmp",IMREAD_COLOR);
    Mat imageCity2 = imread("D:\\An III sem II\\PI\\FilterProject\\images\\city2.bmp",IMREAD_COLOR);

    /* Mat sharpImg = mySharpening(image,1);
    Mat sharpOpenCv = sharpeningOpenCv(image,1);
    Mat emboss = myEmboss(image,0);*/

    imshow("original",image);
    //imshow("sharppening",sharpImg);
    //imshow("openCV",sharpOpenCv);
    //imshow("emboss",emboss);
    //imshow("sepia",sepia(image));
    //imshow("negativ",negativ(image));
    imshow("fisheye2 1",myFisheye2(image,1));
    imshow("fisheye2 1.5",myFisheye2(image,1.5));
    imshow("fisheye2 3",myFisheye2(image,3));
    imshow("fisheye 1",myFisheye(image,1));
    imshow("fisheye 1.5",myFisheye(image,1.5));
    imshow("fisheye 3",myFisheye(image,3));
    //imshow("fisheye2",myFisheye2(imageCity,3));
    //imshow("fisheye",myFisheye(imageCity,3));
    //imshow("tilt shift", tiltShift(image,0.8, 30,100));
    //imshow("tilt shift2", tiltShift(image,1.2, 30,100));

    //la toate functiile sa aplic si implementarea din openCV , iar la partea de testare sa le compar cele doua penru noise
    //pentru comparare folosesc PSNR, SSIM, and LPIPS
    //imshow("original city", imageCity2);
    //imshow("tilt shift",tiltShift(imageCity,1.2, 30,100));
    //imshow("bokeh",bokehWithLights(imageCity,21,50));
    //imshow("bokeh circular",bokehLightEffect(imageCity,180,25));
    //imshow("bokeh 2",bokehEffect(imageCity2,10));
    //labels bfsLabels = BFS_labeling(imageCity);
    //Mat result_bfs = color_labels(bfsLabels);
    //imshow("BFS", result_bfs);
    //imshow("bokeh",bokehCerc(imageCity));
    //printf("%d",bfsLabels.no_labels);
    waitKey(0);
    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.