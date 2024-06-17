#include "FoodLeftoverEstimation.h"



std::vector<std::pair<int, int>> FoodLeftoverEstimation::getLefoverSize(cv::Mat image) {

    cv::Mat grayImage;
    if (image.channels() == 3) {
        cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = image.clone();
    }

    int count = 0;
    std::vector<int> pixelCount(256, 0);
    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols; j++) {
            int intensity = static_cast<int>(grayImage.at<uchar>(i, j));
            if (intensity > 0)
            {
                pixelCount[intensity]++;
            }
            if (intensity == 10)
            {
                count++;
            }
        }
    }

    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < pixelCount.size(); i++) {
        if (pixelCount[i] > 0) {
            result.push_back(std::make_pair(i, pixelCount[i]));
        }
    }

    std::sort(result.begin(), result.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first < b.first;
        });

    std::cout << count<<std::endl;

    return result;
}


void FoodLeftoverEstimation::printLeftoverSize(const std::vector<std::pair<int, int>>& Leftover) {
    std::cout << "Class Value: Region Size" << std::endl;
    for (const auto& item : Leftover) {
        std::cout << item.first << ": " << item.second << std::endl;
    }
    std::cout << std::endl;
}


std::vector<std::pair<int, double>> FoodLeftoverEstimation::calculateFoodLeftoverEstimation(cv::Mat beforeMask, cv::Mat afterMask) {
    std::vector<std::pair<int, int>> before_mask_size = getLefoverSize(beforeMask);
    std::vector<std::pair<int, int>> after_mask_size = getLefoverSize(afterMask);
    std::vector<std::pair<int, double>> foodLeftoverEstimation;
    for (const auto& item : before_mask_size)
    {
        int id = item.first;

        auto it = std::find_if(after_mask_size.begin(), after_mask_size.end(), [id](const std::pair<int, int>& p) {
            return p.first == id;
            });
        if (it == after_mask_size.end()) {
            std::cout << "Mask with different ID: " << std::endl;
            return foodLeftoverEstimation;
        }
       
    }

    for (int i=0;i< before_mask_size.size();i++)
    {
        if (before_mask_size[i].first > 0)
        {
            double Ri = static_cast<double>(after_mask_size[i].second) / before_mask_size[i].second;
            foodLeftoverEstimation.push_back(std::make_pair(before_mask_size[i].first, Ri));
        }
        else
        {
            foodLeftoverEstimation.push_back(std::make_pair(before_mask_size[i].first, 0));
        }
    }
    
    return foodLeftoverEstimation;
}


void FoodLeftoverEstimation::printLeftOverEstimation(std::vector<std::pair<int, double>> estimation)
{
    for (const auto& pair : estimation) {
        std::cout << "ID: " << pair.first << ", Leftover Estimation: " << pair.second << std::endl;
    }
}

