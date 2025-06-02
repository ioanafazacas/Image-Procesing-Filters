#include <iostream>
#include <opencv2/opencv.hpp>
#include "effects.h"


using namespace std;
using namespace cv;


bool IsInside(Mat img, int i, int j){
    if(i >= 0 && i < img.rows && j >= 0 && j < img.cols)
        return true;
    return false;
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

            if (srcX >= 0 && srcX < w-1 && srcY >= 0 && srcY < h-1) {
                result.at<Vec3b>(y, x) = source.at<Vec3b>((int)srcY, (int)srcX);
            }

        }
    }

    return result;
}

Mat enhancedFisheye(Mat source, float k) {
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
            //float r_distorted = r_norm*(1+k*r_norm*r_norm) * maxRadius;
            float r_distorted = pow(r_norm, k) * maxRadius;

            float srcX = center.x + r_distorted * cos(theta);
            float srcY = center.y + r_distorted * sin(theta);

            if (srcX >= 0 && srcX < w-1 && srcY >= 0 && srcY < h-1) {
                // Interpolare biliniara
                int x1 = (int)srcX;
                int y1 = (int)srcY;
                int x2 = x1 + 1;
                int y2 = y1 + 1;

                float dx = srcX - x1;
                float dy = srcY - y1;

                Vec3b p1 = source.at<Vec3b>(y1, x1);
                Vec3b p2 = source.at<Vec3b>(y1, x2);
                Vec3b p3 = source.at<Vec3b>(y2, x1);
                Vec3b p4 = source.at<Vec3b>(y2, x2);

                // Interpolare intre cei 4 pixeli vecini
                Vec3b interpolated =
                        p1 * (1-dx) * (1-dy) +  //stanga-sus
                        p2 * dx * (1-dy) +      // dreapta-sus
                        p3 * (1-dx) * dy +      // stânga-jos
                        p4 * dx * dy;           // dreapta-jos

                result.at<Vec3b>(y, x) = interpolated;
                //result.at<Vec3b>(y, x) = source.at<Vec3b>((int)srcY, (int)srcX);
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
            //printf("%f ",kernel.at<double>(x,y));
        }
        printf("\n");
    }
    printf("Gaussian kernel sum %f",sum);
    return kernel;
}

Mat myGaussianBlur(Mat source, float sigma){
    Mat result = Mat::zeros(source.size(), source.type());
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
            result.at<Vec3b>(x, y)[0] = cv::saturate_cast<uchar>(filterPixel[0]);
            result.at<Vec3b>(x, y)[1] = cv::saturate_cast<uchar>(filterPixel[1]);
            result.at<Vec3b>(x, y)[2] = cv::saturate_cast<uchar>(filterPixel[2]);
        }
    }
    return result;
}

Mat tiltShift(Mat source, float sigma, int low, int high){
    Mat effect = Mat::zeros(source.size(), source.type());
    effect = myGaussianBlur(source,sigma);

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

uchar ClampToByte(float value) {
    return static_cast<uchar>(std::min(255.f, std::max(0.f, value)));
}

Mat bokehEffect(const Mat& input, int radius) {
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

Mat myAddWeighted(Mat src1, double alpha, Mat src2, double beta) {

    Mat dst = Mat::zeros(src1.size(), src1.type());

    for (int y = 0; y < src1.rows; ++y) {
        for (int x = 0; x < src1.cols; ++x) {
            Vec3b pixel1 = src1.at<Vec3b>(y, x);
            Vec3b pixel2 = src2.at<Vec3b>(y, x);
            Vec3b& pixelDst = dst.at<Vec3b>(y, x);

            for (int c = 0; c < 3; ++c) {
                float val = pixel1[c] * alpha + pixel2[c] * beta;
                pixelDst[c] = saturate_cast<uchar>(val);
            }
        }
    }

    return dst;
}

Mat bokehCerc(Mat source) {
    Mat blurred;

    // Obține zonele luminoase etichetate
    labels light_labels = BFS_labeling(source);
    Mat result = source.clone();
    Mat result2 = source.clone();

    // Pentru fiecare componentă etichetată se calculeaza centrul si raza
    for (int lbl = 1; lbl <= light_labels.no_labels; lbl++) {
        vector<Point> points;
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                if (light_labels.labels.at<int>(i,j) == lbl) {
                    points.push_back(Point(j,i));
                }
            }
        }

        // Calculează centrul geometric al componentei
        if (points.empty()) continue;

        Point center(0,0);
        for (auto& p : points) {
            center.x += p.x;
            center.y += p.y;
        }
        center.x = center.x / points.size();
        center.y = center.y / points.size();

        // Calculeaza raza aproximativa a unei componente
        float maxRadius = 0.f;
        for (const auto& p : points) {
            float dist = norm(p - center);
            if (dist > maxRadius) maxRadius = dist;
        }
        //printf("raza - %f\n", maxRadius);
        if(maxRadius>15)maxRadius=15;
        // Desenăm cercul blurat în imaginea rezultată
        circle(result, center, maxRadius, light_labels.colors[lbl], -1, LINE_AA);
        //contopim cele doua imagini
        result2 = myAddWeighted(result, 0.75, source, 0.25);
    }
    //aplicam blurarea
    blurred = myGaussianBlur(result2,1.8);
    return blurred;
}

labels BFS_labeling(Mat source){

    Mat labels;
    int rows, cols, no_labels;

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
                Vec3i color=Vec3i(0,0,0);
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
                Vec3b avgColor;
                if (nrComponent > 0) {
                    avgColor[0] = color[0] / nrComponent;
                    avgColor[1] = color[1] / nrComponent;
                    avgColor[2] = color[2] / nrComponent;
                    colors[no_labels] = avgColor;
                }
            }
        }
    }

    return {labels, no_labels, colors};
}

Mat newBokeh(Mat source, float intensity, int radius ) {
    Mat result = source.clone();

    // Creează kernel circular cu margini moi
    int size = radius * 2 + 1;
    Mat kernel(size, size, CV_32FC1, Scalar(0));
    Point center(radius, radius);

    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            float dist = norm(Point(x, y) - center);
            if (dist <= radius)
                kernel.at<float>(y, x) = pow(1.0f - dist / radius, 2.0f);
        }

    normalize(kernel, kernel, 0, 1, NORM_MINMAX);

    // Detectează zone luminoase
    Mat gray;
    cvtColor(source, gray, COLOR_BGR2GRAY);
    threshold(gray, gray, 200, 255, THRESH_BINARY);

    for (int y = radius; y < source.rows - radius; ++y) {
        for (int x = radius; x < source.cols - radius; ++x) {
            if (gray.at<uchar>(y, x) > 0) {
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int nx = x + dx, ny = y + dy;
                        float k = kernel.at<float>(dy + radius, dx + radius);
                        if (nx >= 0 && nx < source.cols && ny >= 0 && ny < source.rows) {
                            Vec3b& dst = result.at<Vec3b>(ny, nx);
                            Vec3b src = source.at<Vec3b>(y, x);
                            for (int c = 0; c < 3; ++c) {
                                dst[c] = saturate_cast<uchar>((1 - k * intensity) * dst[c] + k * intensity * src[c]);
                            }
                        }
                    }
                }
            }
        }
    }
    Mat result2= source.clone();
    Mat blurred = source.clone();
    result2 = myAddWeighted(result, 0.75, source, 0.25);
    blurred = myGaussianBlur(result2,1.8);

    return blurred;
}


Mat bokehImproved(Mat source) {
    Mat blurred;

    // Obține zonele luminoase etichetate
    labels light_labels = BFS_labeling(source);
    Mat result = source.clone();
    Mat result2 = source.clone();

    // Pentru fiecare componentă etichetată se calculeaza centrul si raza
    for (int lbl = 1; lbl <= light_labels.no_labels; lbl++) {
        vector<Point> points;
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                if (light_labels.labels.at<int>(i,j) == lbl) {
                    points.push_back(Point(j,i));
                }
            }
        }

        // Calculează centrul geometric al componentei
        if (points.empty()) continue;

        Point center(0,0);
        for (auto& p : points) {
            center.x += p.x;
            center.y += p.y;
        }
        center.x = center.x / points.size();
        center.y = center.y / points.size();

        // Calculeaza raza aproximativa a unei componente
        float maxRadius = 0.f;
        for (const auto& p : points) {
            float dist = norm(p - center);
            if (dist > maxRadius) maxRadius = dist;
        }
        //printf("raza - %f\n", maxRadius);
        if(maxRadius>15)maxRadius=15;
        // Desenăm cercul blurat în imaginea rezultată
        circle(result2, center, maxRadius, light_labels.colors[lbl], -1, LINE_AA);

        //Desenarea cercului cu intensitate mare in centru si mai slaba la exterior
        int size = maxRadius * 2 + 1;
        Mat kernel(size, size, CV_32FC1, Scalar(0));
        Point center2(maxRadius, maxRadius);

        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x) {
                float dist = norm(Point(x, y) - center2);
                if (dist <= maxRadius)
                    kernel.at<float>(y, x) = pow(1.0f - dist / maxRadius, 2.0f);
            }
        for (int dy = -maxRadius; dy <= maxRadius; ++dy) {
            for (int dx = -maxRadius; dx <= maxRadius; ++dx) {
                int nx = center.x + dx, ny = center.y + dy;
                float k = kernel.at<float>(dy + maxRadius, dx + maxRadius);
                if (nx >= 0 && nx < source.cols && ny >= 0 && ny < source.rows) {
                    Vec3b& dst = result.at<Vec3b>(ny, nx);
                    for (int c = 0; c < 3; ++c) {
                        dst[c] = saturate_cast<uchar>((1 - k ) * dst[c] + k *  light_labels.colors[lbl][c]);
                    }
                }
            }
        }

    }
    Mat resultFinal;
    //contopirea imaginilor
    resultFinal = myAddWeighted(result, 0.6, result2, 0.4);
    //aplicarea blurarii
    blurred = myGaussianBlur(resultFinal,1.8);
    return blurred;
}
