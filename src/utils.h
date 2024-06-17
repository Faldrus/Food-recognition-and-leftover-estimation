#ifndef FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H
#define FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/calib3d.hpp>

//Function that creates and returns a Mat vector, contains the images for each tray
std::vector<cv::Mat> createVecImgFromSource(std::string path);

//Function that returns a vector with all the labels
std::vector<std::string> getLabels();

//Function to compare colors, it is needed in findMostFrequentColors
bool compareColors(const std::pair<cv::Vec3b, int>& color1, const std::pair<cv::Vec3b, int>& color2);

//Function that finds using the histogram the most frequent colors
std::vector<cv::Vec3b> findMostFrequentColors(const cv::Mat& image, int numColors);

//Function that removes the most frequent colors from the image
cv::Mat removeColors(const cv::Mat& image1,int size,int numColors,int delta);

//function that remove a color with a certain threshold
void removeSimilarPixels(cv::Mat& image, const cv::Scalar& targetColor, int delta);

//Function that return an image inside a box
cv::Mat drawBox(cv::Mat image,cv::Scalar color,cv::Rect& boundingrect);

//Function that prints the values of a rect (x,y,height,width)
void showValuesRectangle(cv::Rect box);

//Not used
cv::Mat getHisto(cv::Mat image);
double compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2);

#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_UTILS_H
