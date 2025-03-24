#include <iostream>
#include <opencv2/opencv.hpp>
#include "src/effects.h"

using namespace std;
using namespace cv;

int main() {

    Mat image = imread("D:\\An III sem II\\PI\\FilterProject\\images\\flowers_24bits.bmp",IMREAD_COLOR);
    Mat sharpImg = sharpening(image,1);
    Mat sharpOpenCv = sharpeningOpenCv(image,1);
    Mat emboss = myEmboss(image,0);
    imshow("original",image);
    imshow("sharppening",sharpImg);
    imshow("openCV",sharpOpenCv);
    imshow("emboss",emboss);
    imshow("emboss2",myEmboss(image,5));
    waitKey(0);
    return 0;
}

// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.