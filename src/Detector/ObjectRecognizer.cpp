#include "ObjectRecognizer.h"




ObjectRecognizer::ObjectRecognizer(std::string Path)
{
	folderPath = Path;
	for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
		int label = std::stoi(entry.path().filename().string());
		std::string folder= folderPath+ entry.path().filename().string();
		std::vector<cv::Mat> template_vector;

		for (const auto& food_file : std::filesystem::directory_iterator(folder)) {

			std::string filepath = folder+"\\"+food_file.path().filename().string();
			template_vector.push_back(cv::imread(filepath));
		}

		data.push_back(FoodTemplate(label, template_vector));

	}

}

std::string ObjectRecognizer::recognize(cv::Mat img_to_recognize)
{
	int prediction = recognize_id(img_to_recognize);
	return convert_id(prediction);
}

std::string ObjectRecognizer::recognize_course(cv::Mat img_to_recognize)
{
	int result = recognize_id(img_to_recognize);
	std::string course = "";
	if (result > 5)
	{
		course = "Second Dish";
	}
	else
	{
		course = "Main Dish";
	}

}

int ObjectRecognizer::recognize_id(cv::Mat img_to_recognize)
{
	double max = 0.0;
	int win_label=0;
	std::vector<double> prob_value;
	std::vector<double> best_match_value;
	double score_template_matching = 0.0;
	double score = 0.0;

	for (int i = 0; i < data.size(); i++)
	{
		std::vector<cv::Mat>& v = data[i].get_images();
		for (int j = 0; j < v.size(); j++)
		{
			prob_value.push_back(compute_similarity(img_to_recognize, v[j]));

			if (score_template_matching < similarityScoretemplatematching(img_to_recognize, v[j]))
			{
				score_template_matching = similarityScoretemplatematching(img_to_recognize, v[j]);
			}
		}
		double mean_similarity = compute_mean(prob_value);
		//best_match_value.push_back(mean_similarity);
		best_match_value.push_back(score_template_matching/v.size());


		prob_value.clear();
		//score_template_matching = score_template_matching / v.size();

		if (score_template_matching > 0.58)
		{
			score = ((score_template_matching * 6) + (3 * mean_similarity)) / 9;
		}
		if (score_template_matching < 0.58)
		{
			score = ((score_template_matching * 4) + (2 * mean_similarity)) / 6;
		}

		if (score_template_matching > max)
		{
			max = score_template_matching;
			win_label = data[i].get_label();
		}

		//std::cout << data[i].get_label() << " ==> " << score << "\n";
		//std::cout << score_template_matching << std::endl;


		score_template_matching = 0.0;

		score = 0.0;


	}

	return win_label;
}

double ObjectRecognizer::compute_mean(std::vector<double>& arr)
{
	double sum = 0.0;
	for (int i = 0; i < arr.size(); i++)
	{
		sum = sum + arr[i];
	}
	return sum / arr.size();
}

std::vector<cv::Mat> ObjectRecognizer::calculate_histo(cv::Mat img)
{

	std::vector<cv::Mat> bgrChannels;
	cv::split(img, bgrChannels);

	cv::Mat imgGray;
	cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
	cv::Mat mask = imgGray > 0;

	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange[] = { range };

	cv::Mat histB, histG, histR;
	cv::calcHist(&bgrChannels[0], 1, 0, mask, histB, 1, &histSize, histRange);
	cv::calcHist(&bgrChannels[1], 1, 0, mask, histG, 1, &histSize, histRange);
	cv::calcHist(&bgrChannels[2], 1, 0, mask, histR, 1, &histSize, histRange);

	return std::vector<cv::Mat>{ histB, histG, histR};
}

std::vector<cv::Mat> ObjectRecognizer::calculate_histo_hsv(cv::Mat img)
{

	cv::Mat image_hsv;
	cv::cvtColor(img, image_hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> hsvChannels;
	cv::split(image_hsv, hsvChannels);


	int hBins = 180;
	float hRange[] = { 0, 180 };
	const float* histRange = { hRange };
	cv::Mat histH;
	cv::calcHist(&hsvChannels[0], 1, 0, cv::Mat(), histH, 1, &hBins, &histRange, true, false);
	histH.at<float>(0) = 0.0f;

	int sBins = 256;
	float sRange[] = { 0, 256 };
	const float* histRangeS = { sRange };
	cv::Mat histS;
	cv::calcHist(&hsvChannels[1], 1, 0, cv::Mat(), histS, 1, &sBins, &histRangeS, true, false);

	int vBins = 256;
	float vRange[] = { 0, 256 };
	const float* histRangeV = { vRange };
	cv::Mat histV;
	cv::calcHist(&hsvChannels[2], 1, 0, cv::Mat(), histV, 1, &vBins, &histRangeV, true, false);

	return std::vector<cv::Mat>{histH, histS, histV};
}

void ObjectRecognizer::remove_light(cv::Mat& img, int threshold)
{
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

			if ((pixel[0] > threshold) && (pixel[1] > threshold) && (pixel[1] > threshold))
			{
				img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
			}

		}
	}
}

double ObjectRecognizer::feature_similariy_score(cv::Mat& image1, cv::Mat& image2)
{
	cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	cv::Mat descriptors1, descriptors2;
	sift->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
	sift->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

	cv::BFMatcher matcher(cv::NORM_L2);

	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	double similarityScore = static_cast<double>(matches.size()) / static_cast<double>(std::max(descriptors1.rows, descriptors2.rows));

	return similarityScore;
}

double ObjectRecognizer::compute_similarity(cv::Mat img, cv::Mat img_template)
{
	cv::Mat resized_template_image;
	cv::Mat resized_img;
	int dimension = 300;
	int comparison_method = 0;
	cv::resize(img_template, resized_template_image, cv::Size(dimension, dimension));

	cv::resize(img, resized_img, cv::Size(dimension, dimension));

	remove_light(resized_template_image, 160);

	remove_light(resized_img, 160);

	std::vector<cv::Mat> template_hist = calculate_histo_hsv(resized_template_image);
	std::vector<cv::Mat> img_hist = calculate_histo_hsv(resized_img);

	//Histogram Blue
	cv::Mat template_hist_blue = template_hist[0];
	cv::Mat img_hist_blue = img_hist[0];

	//Histogram Green
	cv::Mat template_hist_green = template_hist[1];
	cv::Mat img_hist_green = img_hist[1];

	//Histogram Red
	cv::Mat template_hist_red = template_hist[2];
	cv::Mat img_hist_red = img_hist[2];

	double hist_sum = (compareHist(template_hist_blue, img_hist_blue, comparison_method) + compareHist(template_hist_green, img_hist_green, comparison_method) + compareHist(template_hist_red, img_hist_red, comparison_method))/3;

	return hist_sum;
}

std::string ObjectRecognizer::convert_id(int id)
{
	std::string output = "";
	switch (id) {
	case 1:
		output = "pasta with pesto";
		break;
	case 2:
		output = "pasta with tomato sauce";
		break;
	case 3:
		output = "pasta with meat sauce";
		break;
	case 4:
		output = "pasta with clams and mussels";
		break;
	case 5:
		output = "pilaw rice with peppers and peas";
		break;
	case 6:
		output = "grilled pork cutlet";
		break;
	case 7:
		output = "fish cutlet";
		break;
	case 8:
		output = "rabbit";
		break;
	case 9:
		output = "seafood salad";
		break;
	case 10:
		output = "beans";
		break;
	case 11:
		output = "basil potatoes";
		break;
	case 12:
		output = " salad";
		break;
	case 13:
		output = "bread";
		break;
	default:
		output = "";
		break;
	}
	return output;
}

double ObjectRecognizer::similarityScoretemplatematching(cv::Mat img, cv::Mat img_template)
{
	//Image resizing in input
	cv::Mat resized_template_image;
	cv::Mat resized_img;
	int dimension = 300;
	int comparison_method = 0;
	cv::resize(img_template, resized_template_image, cv::Size(img.cols, img.rows));

	cv::resize(img, resized_img, cv::Size(img.cols, img.rows));

	cv::Mat result;
	cv::matchTemplate(resized_img, resized_template_image, result, cv::TM_CCOEFF_NORMED);

	double minVal, maxVal;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	double similarityScore = (maxVal + 1.0) / 2.0;

	return similarityScore;
}



