#include "FoodTemplate.h"


FoodTemplate::FoodTemplate(){}


FoodTemplate::FoodTemplate(int label_name, std::vector<cv::Mat> template_vector)
{
	template_label = label_name;
	template_images = template_vector;
}

int FoodTemplate::get_label()
{
	return template_label;
}

std::vector<cv::Mat>& FoodTemplate::get_images()
{
	return template_images;
}




