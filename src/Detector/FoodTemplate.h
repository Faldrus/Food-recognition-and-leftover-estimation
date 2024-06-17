#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


class FoodTemplate
{
public:

	//Constructor
	FoodTemplate();

	//Constructor that create a new FoodTemplate with a label name and a vector of images for that template
	FoodTemplate(int label_name, std::vector<cv::Mat> template_vector);

	//Return the label of the FoodTemplate
	int get_label();

	//Return the images associated of the FoodTemplate
	std::vector<cv::Mat>& get_images();


private:

	//Label of the FoodTemplate
	int template_label;

	//Vector of images of FoodTemplate
	std::vector<cv::Mat> template_images;


};

