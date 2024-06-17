#pragma once
#include <vector>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>



class mIOU
{
public:
	
	//Calculate the IOU for the classes. If the image is not with 1 channel the method convert it.
	std::vector<std::pair<int, double>> calculateIOUForClasses(const cv::Mat& img_GT, const cv::Mat& img_PR);

	//Print the Iou vector for classes
	void printIOUVector(const std::vector<std::pair<int, double>>& IOU_for_classes);

private:

	//Detect the different ID from the mask given by parameter
	std::vector<int> getClassValues(cv::Mat img);

	//Print the class value from the vector given by parameter.
	void printClassValues(const std::vector<int>& classValues);

	//Calculate the areas of the region from the mask image given by parameter.
	std::vector<std::pair<int, int>> getRegionSize(cv::Mat img);

	//Print the region area of pair given by parameter.
	void printRegionSize(const std::vector<std::pair<int, int>>& regionSize);

	//Calculate the intersection of two mask images. If the image is not with 1 channel the method convert it.
	std::vector<std::pair<int, int>> calculateIntersection(const cv::Mat& img1, const cv::Mat& img2);

	//Print the Intersection of pair given by parameter.
	void printIntersection(const std::vector<std::pair<int, int>>& intersectionCoordinates);

	//Calculate the union of two mask images. If the image is not with 1 channel the method convert it.
	std::vector<std::pair<int, int>> calculateUnion(const cv::Mat& img_GT, const cv::Mat& img_PR);

	//Print the Union of pair given by parameter.
	void printUnionAreas(const std::vector<std::pair<int, int>>& unionAreas);
};

