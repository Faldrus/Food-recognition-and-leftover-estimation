#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "FoodTemplate.h"
#include <opencv2/dnn.hpp>


class ObjectRecognizer
{
public:

	//Constructor that receive by parameter the where tha data template are stored
	ObjectRecognizer(std::string Path);

	//Given an an image in input, 
	std::string recognize_course(cv::Mat img_to_recognize);

	//Given an an image in input, recognize what type of food are from the image.
	std::string recognize(cv::Mat img_to_recognize);

	//Given an an image in input, recognize the id of food inside the image.
	int recognize_id(cv::Mat img_to_recognize);


	
private:
	
	//Compute a similarity value based on the histogram of the image
	double compute_similarity(cv::Mat img, cv::Mat img_template);

	//Compute the men of the vector arr passed throw parameter
	double compute_mean(std::vector<double>& arr);

	//Calculate the histogram of the img and return a vector of 3 channels histogram(BGR)
	std::vector<cv::Mat> calculate_histo(cv::Mat img);

	//Calculate the histogram of the img and return a vector of 3 channels histogram(HSV)
	std::vector<cv::Mat> calculate_histo_hsv(cv::Mat img);

	//Remove the pixel above some threshold.(make it dark)
	void remove_light(cv::Mat& img, int threshold);

	//Calculate a similarity score based on the matching of the SIFT features of image1,image2
	double feature_similariy_score(cv::Mat& image1, cv::Mat& image2);
	void  selectionSort(std::vector<double>& value_vector, std::vector<FoodTemplate>& food_template_vector);

	//Return the similarity matching score between the two images
	double similarityScoretemplatematching(cv::Mat image, cv::Mat template_image);

	//Convert the dish ID with the real dish name
	std::string convert_id(int id);

	//Path where the folder data is stored
	std::string folderPath;

	//Vector of FoodTemplate data where are stored all the template
	std::vector<FoodTemplate> data;
};

