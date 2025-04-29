#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/effects.h"

using namespace std;
using namespace cv;

int main() {

    Mat image = imread("D:\\An III sem II\\PI\\FilterProject\\images\\flowers_24bits.bmp",IMREAD_COLOR);
    Mat sharpImg = mySharpening(image,1);
    Mat sharpOpenCv = sharpeningOpenCv(image,1);
    Mat emboss = myEmboss(image,0);
    Mat fisheye = myFisheye(image,-1);
    Mat fisheyeC = fisheyeCircular(image,1.5);
    imshow("original",image);
    /*imshow("sharppening",sharpImg);
    imshow("openCV",sharpOpenCv);
    imshow("emboss",emboss);
    imshow("fisheye",fisheye);
    imshow("fisheyeCircular",fisheyeC);*/
    imshow("tilt shift", tiltShift(image,0.8, 30,100));
    imshow("tilt shift2", tiltShift(image,1.2, 30,100));
    //la toate functiile sa aplic si implementarea din openCV , iar la partea de testare sa le compar cele doua penru noise
    //pentru comparare folosesc PSNR, SSIM, and LPIPS

    waitKey(0);
    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.