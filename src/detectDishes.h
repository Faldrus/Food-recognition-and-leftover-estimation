#ifndef FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H
#define FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//Function that given an image, returns a new image with only the salad in the picture
cv::Mat detectSalad(const cv::Mat& image);

//Function that given an image, returns a new image with only the dishes in the picture
cv::Mat detectDishesEdge(const cv::Mat& image);

//Function that given an image, returns a new image with dishes and salad in the picture
cv::Mat detectFoods(const cv::Mat& image,int n);

//Function that finds bread based on histogram and morphological operations and colors
cv::Mat detectBreadByHisto(const cv::Mat& image,int n);

//Function that removes background elements around the desired object
cv::Mat removeDishes(cv::Mat image, int delta);

//Function that gets the yogurt mask to remove it
cv::Mat getMaskYogurt(const cv::Mat& image);

//Function that removes dishes, yogurt, salad from the image
cv::Mat preparePhoto(const cv::Mat& image,int n);

//Not used
cv::Mat allF(const cv::Mat& img);

//Function that returns a vector containing individual dish from an image with multiple plates
std::vector<cv::Mat> getOneDish(const cv::Mat &image);

//Function to segment the first dish
cv::Mat segmentFirst(cv::Mat image);

//Function that calls segmentFirst and sets the pixels outside the mask to zero.
//Used to perform the final form of segmentation for first dish
cv::Mat getFirst(cv::Mat image);

//Function that returns a vector containing the individual dishes in each tray
//set value of i and j to work on specific trays
std::vector<cv::Mat> getFoodImageByAllTrays(std::vector<std::vector<cv::Mat>> trays,int nTray);

//Function that segment the salad
cv::Mat segmentSalad(cv::Mat image);

//Function to set the radius for tray4 (works as detectDishesEdges)
cv::Mat detectDishesEdge4(const cv::Mat& image);

//Function to set the radius for tray4 (works as detectOneDish)
std::vector<cv::Mat> getOneDish4(const cv::Mat &image);

//Function to segment the second dish
cv::Mat segmentSecond(const cv::Mat& image);

//Function that calls segmentSecond and sets the pixels outside the mask to zero.
//Used to perform the final form of segmentation for second dish
cv::Mat getSecond(const cv::Mat& image);

//Kmeans
cv::Mat K_Means(cv::Mat image, int k);

//Function that displays colored boxes around various dishes, salad, bread (if any), also allows access to images of individual
//food inside the box and box values (coordinates, height,width)
std::vector<cv::Mat> viewBoxOnImage(std::vector<cv::Mat>firstTray,std::vector<cv::Mat>saladTray,std::vector<cv::Mat>secondTray,std::vector<cv::Mat>breadTray,std::vector<cv::Mat>tray,std::vector<cv::Mat>output,std::vector<cv::Mat>&boxFirst,std::vector<cv::Mat>&boxSecond,std::vector<cv::Mat>&boxSalad,std::vector<cv::Mat>&boxBread,std::vector<cv::Rect>&boxRectFirst,std::vector<cv::Rect>&boxRectSecond,std::vector<cv::Rect>&boxRectSalad,std::vector<cv::Rect>&boxRectBread,int n);

//Funcation to get all the masks of an image
cv::Mat getAllMask(cv::Mat img,int nTray);

#endif //FOOD_RECOGNITION_AND_LEFTOVER_ESTIMATION_DETECTDISHES_H

