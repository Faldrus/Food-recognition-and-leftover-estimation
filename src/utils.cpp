#include "utils.h"
using namespace std;
using namespace cv;

//Function that creates and returns a Mat vector, contains the images for each tray
vector<Mat> createVecImgFromSource(string path)
{
    vector<Mat> images;
    vector<string> fileNames;
    glob(path, fileNames);
    for (const auto& filename : fileNames)
    {
        Mat image = imread(filename);
        images.push_back(image);
    }
    return images;

}

//Function to compare colors, it is needed in findMostFrequentColors
bool compareColors(const pair<Vec3b, int>& color1, const pair<Vec3b, int>& color2)
{
    return color1.second > color2.second;
}

//Function that finds using the histogram the most frequent colors
vector<Vec3b> findMostFrequentColors(const Mat& image, int numColors)
{
    vector<Vec3b> mostFrequentColors;
    vector<Mat> channels;
    split(image, channels);

    int histSize = 256;
    float range[] = {1, 256};
    const float* histRange = {range};
    bool uniform = true;
    bool accumulate = false;
    Mat b_hist, g_hist, r_hist;
    calcHist(&channels[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&channels[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    vector<pair<Vec3b, int>> colorFrequencies;

    for (int i = 1; i < histSize; i++)
    {
        Vec3b color(i, i, i);
        int frequency = b_hist.at<float>(i) + g_hist.at<float>(i) + r_hist.at<float>(i);

        if (color != Vec3b(0, 0, 0))
        {
            colorFrequencies.push_back(make_pair(color, frequency));
        }
    }

    sort(colorFrequencies.begin(), colorFrequencies.end(), compareColors);

    int colorsToSelect = min(numColors, static_cast<int>(colorFrequencies.size()));
    for (int i = 0; i < colorsToSelect; i++)
        mostFrequentColors.push_back(colorFrequencies[i].first);

    return mostFrequentColors;
}

//Function that removes the most frequent colors from the image
Mat removeColors(const Mat& image1,int size,int numColors,int delta)
{
    Mat image;
    image1.copyTo(image);
    vector<Vec3b> mostFrequentColors = findMostFrequentColors(image, numColors);
    for(int i=0;i<size;i++)
    {
        Vec3b targetColor = mostFrequentColors[i];
        for (int j = 0; j < image.rows; j++)
        {
            for (int k = 0; k < image.cols; k++)
            {
                Vec3b currentColor = image.at<Vec3b>(j, k);
                if (abs(currentColor[0] - targetColor[0]) <= delta && abs(currentColor[1] - targetColor[1]) <= delta && abs(currentColor[2] - targetColor[2]) <= delta)
                    image.at<Vec3b>(j, k) = Vec3b(0, 0, 0);
            }
        }
    }
    return image;
}

//Function that returns a vector with all the labels
vector<string> getLabels()
{
    vector<string> labels = {
            "Background",
            "pasta with pesto",
            "pasta with tomato sauce",
            "pasta with meat sauce",
            "pasta with clams and mussels",
            "pilaw rice with peppers and peas",
            "grilled pork cutlet",
            "fish cutlet",
            "rabbit",
            "seafood salad",
            "beans",
            "basil potatoes",
            "salad",
            "bread"
    };
    return labels;

}

//function that remove a color with a certain threshold
void removeSimilarPixels(Mat& image, const Scalar& targetColor, int delta)
{
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Vec3b pixel = image.at<Vec3b>(i, j);
            int diff = 0;
            for (int k = 0; k < 3; k++)
                diff += abs(pixel[k] - targetColor[k]);

            if (diff <= delta)
            {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
            }
            image.at<Vec3b>(i, j) = pixel;
        }
    }
}

//Function that return an image inside a box
Mat drawBox(Mat image,Scalar color,Rect& boundingBox)
{
    Mat grayImage = image.clone();
    cvtColor(image, grayImage, COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    vector<Vec4i> details;
    findContours(grayImage, contours, details, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Rect boundingRect;
    for (const auto& contour : contours)
    {
        Rect currentRect = cv::boundingRect(contour); // need cv:: to use it
        if (boundingRect.empty())
            boundingRect = currentRect;
        else
            boundingRect = boundingRect | currentRect;
    }

    boundingBox = boundingRect;

    Mat output = image.clone();
    rectangle(output, boundingRect, color, 3);



    return output;
}

//Function that prints the values of a rect (x,y,height,width)
void showValuesRectangle(Rect box)
{
    int x,y,width,height;
    x = box.x;
    y = box.y;
    width = box.width;
    height = box.height;
    cout<<"x:"<<x<<" y:"<<y<<" width:"<<width<<" height:"<<height<<endl;
}

//Not used
Mat getHisto(Mat img)
{
    Mat image = img.clone();
    cvtColor(image,image,COLOR_BGR2GRAY);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    Mat histogram;
    calcHist(&image, 1, 0, Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);

    int histWidth = 512;
    int histHeight = 400;
    int binWidth = cvRound((double)histWidth / histSize);

    Mat histImage(histHeight, histWidth, CV_8UC1, Scalar(0));
    normalize(histogram, histogram, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < histSize; i++)
        line(histImage, cv::Point(binWidth * (i - 1), histHeight - cvRound(histogram.at<float>(i - 1))),Point(binWidth * (i), histHeight - cvRound(histogram.at<float>(i))),Scalar(255), 2, 8, 0);

    return histImage;
}
double compareHistograms(const Mat& hist1, const Mat& hist2)
{

    Mat normalizedHist1, normalizedHist2;
    normalize(hist1, normalizedHist1, 0, 1, NORM_MINMAX, CV_32F);
    normalize(hist2, normalizedHist2, 0, 1, NORM_MINMAX, CV_32F);

    double result = compareHist(normalizedHist1, normalizedHist2, HISTCMP_CORREL);

    return result;
}