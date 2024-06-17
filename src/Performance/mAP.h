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
#include <filesystem>

class mAP
{

public:

	//Compute the intersection between two map composed by the id and the rectangle coordinates
	std::map<int, std::vector<int>> computeIntersection(const std::map<int, std::vector<int>>& map1,
		const std::map<int, std::vector<int>>& map2);

	//Compute the area of the rectangle composed by a map(ID,rectangle coordinates)
	std::vector<int> getArea(const std::map<int, std::vector<int>>& map);

	//Claculate the IOU of the given map composed by ID and rectangle coordinates
	std::vector<std::pair<int, float>> computeIoU(const std::map<int, std::vector<int>>& gtbb, const std::map<int, std::vector<int>>& pred);

	//Method that print the IOU vector
	void printIoU(const std::vector<std::pair<int, float>>& iouVector);

	//Method that print a map composed by ID and by rectangle coordinates
	void printMap(const std::map<int, std::vector<int>>& mappa);

	// get the matches (TP; FP ; TN; FN) depending on the IoU value associated to each ID
	std::vector<std::pair<int, std::string>> classifyMatches(const std::vector<std::pair<int, float>>& iouData);

	//Print the matches passed by parameter
	void printMatches(const std::vector<std::pair<int, std::string>>& matches);

	//Calculate the cumulativeTP based on the prediction passed in input
	std::vector<std::pair<int, int>> calculateCumulativeTP(const std::vector<std::pair<int, std::string>>& predictions);

	//Print the cumulativeTP of the vector passed in input
	void printCumulativeTP(const std::vector<std::pair<int, int>>& cumulativeTP);

	//Calculate the cumulativeFP based on the prediction passed in input
	std::vector<std::pair<int, int>> calculateCumulativeFP(const std::vector<std::pair<int, std::string>>& predictions);

	//Print the cumulativeFP of the vector passed in input
	void printCumulativeFP(const std::vector<std::pair<int, int>>& cumulativeFP);

	//Calculate the Precision of the system based on true positive and false positive.(P = TP / (TP + FP))
	std::vector<std::pair<int, float>> calculatePrecision(const std::vector<std::pair<int, int>>& truePositives, const std::vector<std::pair<int, int>>& falsePositives);

	//Print the precision of the vector passed in input
	void printPrecision(const std::vector<std::pair<int, float>>& precisionValues);

	//Calculate the Recall of the system based on comulativeTP and Ground Throw.(R = TP / Total Ground Truths)
	std::vector<std::pair<int, float>> calculateRecall(const std::vector<std::pair<int, int>>& cumulativeTruePositives, float totalGroundTruths);

	//Print the recall of the vector passed in input
	void printRecall(const std::vector<std::pair<int, float>>& recallValues);

	//Calculate the interpolationPrecision(AP)
	std::vector<std::pair<int, float>> interpolatePrecision(const std::vector<std::pair<int, float>>& precisionValues, const std::vector<std::pair<int, float>>& recallValues);

	//Print the interpolationPrecision(AP) 
	void printInterpolatedPrecision(std::vector<std::pair<int, float>> interpolatedPrecision);

	void printIoUValues(const std::vector<std::pair<int, float>>& iouValues);

	

	

	
	
private:

};

