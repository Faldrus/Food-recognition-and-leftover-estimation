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


class FoodLeftoverEstimation
{
public:

	//Calculate the Lefotver Estimation given the beforeMask and the afterMask Mat images.
	std::vector<std::pair<int, double>> calculateFoodLeftoverEstimation(cv::Mat beforeMask, cv::Mat afterMask);

	// Print the the estimation obtained from the calculateFoodLeftoverEstimation() method with the estimation vector of pair in input.
	void printLeftOverEstimation(std::vector<std::pair<int, double>> estimation);

private:

	//Calculate the number of pixel for every id of the image received in input.
	std::vector<std::pair<int, int>> getLefoverSize(cv::Mat image);

	// Print the the number of pixel and the id returned from getLeftoverSize() method with the leftover vector of pair in input.
	void printLeftoverSize(const std::vector<std::pair<int, int>>& Leftover);
};

