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

Mat sepia(Mat source){
    Mat result= Mat(source.size(), source.type());
    for(int i=0 ; i< source.rows; i++){
        for(int j=0; j< source.cols; j++){
            Vec3b pixel= source.at<Vec3b>(i,j);
            uchar b= source.at<Vec3b>(i,j)[0];
            uchar g= source.at<Vec3b>(i,j)[1];
            uchar r= source.at<Vec3b>(i,j)[2];
            result.at<Vec3b>(i,j)[0]=cv::saturate_cast<uchar>(0.272*r+0.534*g+0.131*b);
            result.at<Vec3b>(i,j)[1]=cv::saturate_cast<uchar>(0.349*r+0.686*g+0.168*b);
            result.at<Vec3b>(i,j)[2]=cv::saturate_cast<uchar>(0.393*r+0.769*g+0.189*b);
        }
    }
    return result;
}

Mat negativ(Mat source){
    Mat result= Mat(source.size(), source.type());
    for(int i=0 ; i< source.rows; i++){
        for(int j=0; j< source.cols; j++){
            Vec3b pixel= source.at<Vec3b>(i,j);
            result.at<Vec3b>(i,j)[0]=255-pixel[0];
            result.at<Vec3b>(i,j)[1]=255-pixel[1];
            result.at<Vec3b>(i,j)[2]=255-pixel[2];
        }
    }
    return result;
}

Mat myFisheye(Mat source, float k){
    Mat result = Mat::zeros(source.size(), source.type());
    float xn, yn;
    double r;
    float scaleX = source.cols/2.0f;
    float scaleY = source.rows/2.0f;
    //center
    float originX=source.cols/2.0f;
    float originY=source.rows/2.0f;

    for(int y=0; y<source.rows; y++){
        for(int x=0; x<source.cols; x++){
            //normalize x,y
            xn=(float)(x-originX)/scaleX;
            yn=(float)(y-originY)/scaleY;
            r=sqrt(xn*xn+yn*yn);

            float theta= atan2(yn,xn);

            //for the circular effect
            float cornerScale = min(abs(1/sin(theta)), abs(1/cos(theta)));
            if(r<1)cornerScale=1;

            float r_distorted = pow(r, k) * cornerScale;

            float srcX = originX + scaleX * r_distorted * cos(theta);
            float srcY = originY + scaleY * r_distorted * sin(theta);

            if (srcX >= 0 && srcX < source.cols && srcY >= 0 && srcY < source.rows) {
                result.at<Vec3b>(y, x) = source.at<Vec3b>((int)srcY, (int)srcX);
            }
        }
    }

    return result;
}



Mat myFisheye2(Mat source, float k) {
    int w = source.cols;
    int h = source.rows;
    Point2f center(w / 2.0f, h / 2.0f);
    float maxRadius = min(w, h) / 2.0f;

    Mat result = Mat::zeros(source.size(), source.type());

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float dx = x - center.x;
            float dy = y - center.y;
            float r = sqrt(dx * dx + dy * dy);

            float theta = atan2(dy, dx);
            float r_norm = r / maxRadius;
            float r_distorted = pow(r_norm, k) * maxRadius;

            float srcX = center.x + r_distorted * cos(theta);
            float srcY = center.y + r_distorted * sin(theta);

            if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h) {
                result.at<Vec3b>(y, x) = source.at<Vec3b>((int)srcY, (int)srcX);
            }

        }
    }

    return result;
}

Mat createGaussianKernel(float sigma){
    int w= round(sigma * 6);
    if(w%2==0)w=w+1;
    printf("w = %d\n",w);
    printf("w/2 = %d\n",w/2);
    Mat kernel = Mat(w,w,CV_64FC1);
    double sum=0;
    for(int x=-w/2; x<=w/2; x++){
        for(int y=-w/2; y<=w/2; y++){
            kernel.at<double>(x+w/2,y+w/2)=exp(-(float)(x*x+y*y)/(2*sigma*sigma))/(2*CV_PI*sigma*sigma);
            sum = sum+kernel.at<double>(x+w/2,y+w/2);
        }
    }
    for(int x=0; x<w; x++){
        for(int y=0; y<w; y++){
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
    int w= round(sigma * 6);
    if(w%2==0)w=w+1;
    for(int x=0; x<source.rows; x++){
        for(int y=0; y<source.cols; y++){
            Vec3d filterPixel= Vec3d (0,0,0);
            for(int i=-w/2; i<=w/2; i++){
                for(int j=-w/2; j<=w/2; j++){
                    if(IsInside(source,x+i,y+j)){
                        double k= kernel.at<double>(i+w/2,j+w/2);
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
//....................................................................................
Mat addArtificialBokehLights(Mat img, int count = 50) {
    RNG rng;
    Mat overlay = img.clone();

    for (int i = 0; i < count; ++i) {
        // Coordonate aleatoare
        int x = rng.uniform(0, img.cols);
        int y = rng.uniform(0, img.rows);

        // Culoare și mărime aleatorii
        int radius = rng.uniform(5, 25);
        Scalar color = Scalar(rng.uniform(180, 255), rng.uniform(180, 255), rng.uniform(180, 255));

        // Desenează un cerc transparent (lumină blurată)
        circle(overlay, Point(x, y), radius, color, -1, LINE_AA);
    }

    // Aplicați un blur ușor pentru estomparea luminilor
    GaussianBlur(overlay, overlay, Size(21, 21), 0);

    // Combinăm cu imaginea originală (alpha blending)
    addWeighted(overlay, 0.3, img, 0.7, 0, img);

    return img;
}

Mat bokehWithLights(Mat source, int blurStrength, int lightCount) {
    // 1. Gaussian Blur pentru fundal
    Mat blurred;
    GaussianBlur(source, blurred, Size(blurStrength | 1, blurStrength | 1), 0);

    // 2. Mască circulară
    Mat mask = Mat::zeros(source.size(), CV_8UC1);
    Point center(source.cols / 2, source.rows / 2);
    int radius = std::min(source.cols, source.rows) / 4;
    circle(mask, center, radius, Scalar(255), -1);

    // 3. Combinație: zona clară + zona blurată
    Mat result = Mat::zeros(source.size(), source.type());
    source.copyTo(result, mask);
    blurred.copyTo(result, 255 - mask);

    // 4. Adaugă luminile artificiale în zona blurată
    result = addArtificialBokehLights(result, lightCount);

    return result;
}

Mat circularBokehKernel(int size) {
    Mat kernel = Mat::zeros(size, size, CV_32F);
    Point center = Point(size / 2, size / 2);
    float radius = size / 2.0;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float dist = norm(Point(i, j) - center);
            if (dist <= radius) {
                kernel.at<float>(i, j) = 1.0;
            }
        }
    }

    // Normalizează kernelul
    kernel /= sum(kernel)[0];
    return kernel;
}

Mat bokehLightEffect(Mat img, int thresholdValue, int kernelSize) {
    Mat gray, lightMask;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 1. Detectează luminile (zonele albe intense)
    threshold(gray, lightMask, thresholdValue, 255, THRESH_BINARY);

    // 2. Kernel circular pentru bokeh
    Mat bokehKernel = circularBokehKernel(kernelSize);

    // 3. Aplica blur pe luminile extrase
    Mat lightBlur, lightsOnly;
    img.copyTo(lightsOnly, lightMask);  // extrage doar luminile
    filter2D(lightsOnly, lightBlur, -1, bokehKernel);

    // 4. Combină cu imaginea originală
    Mat result = img.clone();
    lightBlur.copyTo(result, lightMask);  // înlocuiește luminile cu versiunea blurată

    return result;
}
//.....................................................................................................
uchar ClampToByte(float value) {
    return static_cast<uchar>(std::min(255.f, std::max(0.f, value)));
}

Mat bokehEffect(const Mat& input, int radius) {
    //CV_Assert(input.type() == CV_8UC3);
    Mat output = Mat::zeros(input.size(), input.type());

    int width = input.cols;
    int height = input.rows;
    float radiusSq = radius * radius;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3f sum = Vec3f(0, 0, 0);
            float weightSum = 0.0f;

            for (int ny = std::max(0, y - radius); ny < std::min(height, y + radius); ny++) {
                for (int nx = std::max(0, x - radius); nx < std::min(width, x + radius); nx++) {
                    float dx = static_cast<float>(nx - x);
                    float dy = static_cast<float>(ny - y);
                    float distSq = dx * dx + dy * dy;

                    if (distSq <= radiusSq) {
                        Vec3b pixel = input.at<Vec3b>(ny, nx);
                        float brightness = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;

                        // Pixelii mai luminoși contribuie mai mult (efect bokeh)
                        float weight = sqrtf(brightness + 1.0f);

                        sum[0] += pixel[0] * weight;
                        sum[1] += pixel[1] * weight;
                        sum[2] += pixel[2] * weight;

                        weightSum += weight;
                    }
                }
            }

            Vec3b& outPixel = output.at<Vec3b>(y, x);
            outPixel[0] = ClampToByte(sum[0] / (weightSum + 1e-5f));
            outPixel[1] = ClampToByte(sum[1] / (weightSum + 1e-5f));
            outPixel[2] = ClampToByte(sum[2] / (weightSum + 1e-5f));
        }
    }

    return output;
}
//.......................................................................................

Mat bokehCerc(Mat source, int radius){
    Mat result = Mat::zeros(source.size(), source.type());
    for(int i=0; i<source.rows; i++){
        for(int j=0; j<source.cols; j++){
            Vec3b pixel = source.at<Vec3b>(i,j);
            float brightness = (pixel[0]+pixel[1]+pixel[2])/3.0f;
            if(brightness>180){

            }
        }
    }
    return result;
}

labels BFS_labeling(Mat source){

    /*
     * This method will implement the BFS labeling algorithm
     * Hint:
     *  Use the Point structure(or a similar one) to store the coordinates in a queue
     *  You can use queue from C++ with its specific actions (push, pop, empty, front)
     */
    Mat labels;
    int rows, cols, no_labels;

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    no_labels=0;
    rows=source.rows;
    cols=source.cols;

    labels = Mat::zeros(rows, cols, CV_32SC1);
    vector<Vec3b> colors(10000);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Vec3b pixel = source.at<Vec3b>(i,j);
            float brightness = (pixel[0]+pixel[1]+pixel[2])/3.0f;
            if (brightness>160 && labels.at<int>(i, j) == 0) {
                no_labels++;
                queue<Point> Q;
                labels.at<int>(i, j) = no_labels;
                Q.push(Point(j, i));
                Vec3b color=Vec3b(0,0,0);
                int nrComponent=0;
                while (!Q.empty()) {
                    Point q = Q.front();
                    Q.pop();
                    color[0] = color[0]+ source.at<Vec3b>(q.y, q.x)[0];
                    color[1] = color[1]+ source.at<Vec3b>(q.y, q.x)[1];
                    color[2] = color[2]+ source.at<Vec3b>(q.y, q.x)[2];
                    nrComponent++;
                    for (int k = 0; k < 8; k++) {
                        int ni = q.y + n8_di[k];
                        int nj = q.x + n8_dj[k];

                        if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                            Vec3b neighbour = source.at<Vec3b>(ni, nj);
                            float brightnessNeighbour = (neighbour[0]+neighbour[1]+neighbour[2])/3.0f;
                            if (brightnessNeighbour>160 && labels.at<int>(ni, nj) == 0) {
                                labels.at<int>(ni, nj) = no_labels;
                                Q.push(Point(nj, ni));
                            }
                        }
                    }
                }
                printf("color - %d %d %d\n",color[0],color[1],color[2]);
                color[0]=color[0]/nrComponent;
                color[1]=color[1]/nrComponent;
                color[2]=color[2]/nrComponent;
                colors[no_labels]=color;
            }
        }
    }

    return {labels, no_labels, colors};
}

Mat color_labels(labels labels_str){

    /*
     * This method will generate a number of no_labels colors and
     * generate a color image containing each label displayed in a different color
     */

    int rows, cols, no_labels;
    Mat labels, result;
    Vec3b* colors;

    //*****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    srand(time(NULL));
    no_labels=labels_str.no_labels;
    rows = labels_str.labels.rows;
    cols = labels_str.labels.cols;
    labels = labels_str.labels;
    result = Mat::zeros(rows,cols,CV_8UC3);
    colors=(Vec3b*)calloc(no_labels+1,sizeof(Vec3b));
    colors[0]= Vec3b(255,255,255);
    for(int i=1; i<=no_labels; i++)
    {
        colors[i] = Vec3b(rand()%255,rand()%255,rand()%255);
    }

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            result.at<Vec3b>(i,j)=labels_str.colors[labels.at<int>(i,j)];
        }
    }

    //*****END OF YOUR CODE(DO NOT DELETE / MODIFY THIS LINE) *****

    return result;

}