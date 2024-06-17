#include "mIOU.h"



std::vector<int> mIOU::getClassValues(cv::Mat img) {
    
    cv::Mat imgGray;
    img.copyTo(imgGray);

    cv::cvtColor(imgGray, imgGray, cv::COLOR_BGR2GRAY);
    int cont = 0;
    std::vector<int> classValues;

    for (int i = 0; i < imgGray.rows; i++) {
        for (int j = 0; j < imgGray.cols; j++) {
            int color = imgGray.at<uchar>(i, j);

            if (color > 0 && std::find(classValues.begin(), classValues.end(), color) == classValues.end()) {
                classValues.push_back(color);
            }
            if (color == 10)
            {
                cont++;
            }
        }
    }

    std::sort(classValues.begin(), classValues.end());

    //std::cout << cont;
    return classValues;

}


void mIOU::printClassValues(const std::vector<int>& classValues) {
    std::cout << "Class Values: ";
    for (const auto& value : classValues) {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}


std::vector<std::pair<int, int>> mIOU::getRegionSize(cv::Mat image) {
    
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImage = image.clone();
    }


    std::vector<int> pixelCount(256, 0);
    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols; j++) {
            int intensity = static_cast<int>(grayImage.at<uchar>(i, j));
            if (intensity > 0) 
            { 
                pixelCount[intensity]++;
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



    return result;
}


void mIOU::printRegionSize(const std::vector<std::pair<int, int>>& regionSize) {
    std::cout << "Class Value: Region Size" << std::endl;
    for (const auto& item : regionSize) {
        std::cout << item.first << ": " << item.second << std::endl;
    }
    std::cout << std::endl;
}


std::vector<std::pair<int, int>> mIOU::calculateIntersection(const cv::Mat& img_GT, const cv::Mat& img_PR) {
    CV_Assert(img_GT.size() == img_PR.size());

    cv::Mat image1, image2;
    if (img_GT.channels() != 1) 
    {
        cv::cvtColor(img_GT, image1, cv::COLOR_BGR2GRAY);
    }
    else {
        image1 = img_GT.clone();
    }
    if (img_PR.channels() != 1) 
    {
        cv::cvtColor(img_PR, image2, cv::COLOR_BGR2GRAY);
    }
    else 
    {
        image2 = img_PR.clone();
    }

    std::vector<std::pair<int, int>> intersectionAreas;

    std::vector<int> pixelCounts(256, 0);

    for (int i = 0; i < image1.rows; i++) {
        for (int j = 0; j < image1.cols; j++) {
            int intensity1 = static_cast<int>(image1.at<uchar>(i, j));
            int intensity2 = static_cast<int>(image2.at<uchar>(i, j));

            if (intensity1 == intensity2 && intensity1 != 0) {
                pixelCounts[intensity1]++;
            }
        }
    }

    for (int i = 1; i < pixelCounts.size(); i++) 
    {
        if (pixelCounts[i] > 0) 
        {
            intersectionAreas.push_back(std::make_pair(i, pixelCounts[i]));
        }
    }

    std::vector<int> GT_class = getClassValues(img_GT);
    std::vector<int> pair_class;
    for (int i = 0; i < intersectionAreas.size(); i++)
    {
        pair_class.push_back(intersectionAreas[i].first);
    }
    for (int i = 0; i < GT_class.size(); i++)
    {
        auto iter = std::find(pair_class.begin(), pair_class.end(), GT_class[i]);

        if (iter == pair_class.end()) {
            
            intersectionAreas.push_back(std::make_pair(GT_class[i], 0));
        }
    }

    std::sort(intersectionAreas.begin(), intersectionAreas.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first < b.first;
        });



    return intersectionAreas;
}


void mIOU::printIntersection(const std::vector<std::pair<int, int>>& intersectionCoordinates) {
    std::cout << "Intersections:" << std::endl;
    for (const auto& coord : intersectionCoordinates) 
    {
        std::cout << "ID: " << coord.first << " Areas: " << coord.second << "" << std::endl;
    }
    std::cout<< std::endl;
}


 std::vector<std::pair<int, int>> mIOU::calculateUnion(const cv::Mat & img_GT, const cv::Mat & img_PR) {
        CV_Assert(img_GT.size() == img_PR.size());

        cv::Mat image1, image2;
        if (img_GT.channels() != 1)
        {
            cv::cvtColor(img_GT, image1, cv::COLOR_BGR2GRAY);
        }
        else 
        {
            image1 = img_GT.clone();
        }
        if (img_PR.channels() != 1)
        {
            cv::cvtColor(img_PR, image2, cv::COLOR_BGR2GRAY);
        }
        else 
        {
            image2 = img_PR.clone();
        }

        std::unordered_map<int, int> unionAreas;

        for (int i = 0; i < image1.rows; i++) 
        {
            for (int j = 0; j < image1.cols; j++) 
            {
                int intensity1 = static_cast<int>(image1.at<uchar>(i, j));
                int intensity2 = static_cast<int>(image2.at<uchar>(i, j));

                if (intensity1 != 0 && intensity1 == intensity2) 
                {
                    if (unionAreas.find(intensity1) != unionAreas.end()) 
                    {
                        unionAreas[intensity1] += 1;
                    }
                    else 
                    {
                        unionAreas[intensity1] = 1;
                    }
                }
            }
        }

        std::vector<std::pair<int, int>> unionResults;
        for (const auto& entry : unionAreas) 
        {
            unionResults.push_back(std::make_pair(entry.first, entry.second));
        }

        std::sort(unionResults.begin(), unionResults.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first < b.first;
        });

        return unionResults;
}


void mIOU::printUnionAreas(const std::vector<std::pair<int, int>>& unionAreas) {
    std::cout << "Union Area: \n";
    for (const auto& area : unionAreas) {
        std::cout << "ID: " << area.first << ", Areas: " << area.second << std::endl;
    }
    std::cout << std::endl;
}


std::vector<std::pair<int, double>> mIOU::calculateIOUForClasses(const cv::Mat& img_GT, const cv::Mat& img_PR) {
    std::vector<std::pair<int, double>> iouResults;

    std::vector<int> classValues = getClassValues(img_GT);
    std::vector<std::pair<int, int>> intersectionAreas = calculateIntersection(img_GT, img_PR);
    std::vector<std::pair<int, int>> unionAreas = calculateUnion(img_GT, img_PR);

    for (int classValue : classValues) {
        int intersectionArea = 0;
        int unionArea = 0;

        for (const auto& entry : intersectionAreas) {
            if (entry.first == classValue) {
                intersectionArea = entry.second;
                break;
            }
        }

        for (const auto& entry : unionAreas) {
            if (entry.first == classValue) {
                unionArea = entry.second;
                break;
            }
        }

        double iou = 0.0;
        if (unionArea != 0) {
            iou = static_cast<double>(intersectionArea) / static_cast<double>(unionArea);
        }

        iouResults.push_back(std::make_pair(classValue, iou));
    }

    return iouResults;
}


void mIOU::printIOUVector(const std::vector<std::pair<int, double>>& IOU_for_classes) {
    std::cout << "IOU Value: " << std::endl;
    for (const auto& entry : IOU_for_classes) {
        std::cout << "ID " << entry.first << " IOU: " << entry.second << std::endl;
    }
    std::cout << std::endl;
}