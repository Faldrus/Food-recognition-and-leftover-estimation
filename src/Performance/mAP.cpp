#include "mAP.h"

float IOU_THRESHOLD = 0.5;



std::map<int, std::vector<int>> mAP::computeIntersection(const std::map<int, std::vector<int>>& map_GT,
    const std::map<int, std::vector<int>>& map_PR) {
    std::map<int, std::vector<int>> intersection;

    for (const auto& [key, value] : map_GT) {
        cv::Point up_leftGT = { value[0], value[1] };
        int widthGT = value[2];
        int heightGT = value[3];

        auto it = map_PR.find(key);
        if (it != map_PR.end()) {
            const std::vector<int>& value2 = it->second;
            cv::Point up_leftPred = { value2[0], value2[1] };
            int widthPred = value2[2];
            int heightPred = value2[3];

            cv::Point p1 = { std::max(up_leftGT.x, up_leftPred.x),
                        std::max(up_leftGT.y, up_leftPred.y) };
            cv::Point p2 = { std::min(up_leftGT.x + widthGT, up_leftPred.x + widthPred),
                        std::min(up_leftGT.y + heightGT, up_leftPred.y + heightPred) };

            int iWidth = std::max(p2.x - p1.x + 1, 0);
            int iHeight = std::max(p2.y - p1.y + 1, 0);

            intersection.insert({ key, {p1.x, p1.y, iWidth, iHeight} });
        }
    }

    return intersection;
}

std::vector<int> mAP::getArea(const std::map<int, std::vector<int>>& rectMap) {
    std::vector<int> areas;
    for (const auto& item : rectMap) {
        const std::vector<int>& rect = item.second;
        int area = rect[2] * rect[3];
        areas.push_back(area);
    }
    return areas;
}

void mAP::printMap(const std::map<int, std::vector<int>>& map) {
    for (const auto& [key, value] : map) {
        std::cout << "Object: " << key << ", Value: ";
        for (const auto& element : value) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }
}

void mAP::printIoU(const std::vector<std::pair<int, float>>& iouVector) {
    std::cout << "ID:  IoU" << std::endl;
    for (const auto& item : iouVector) {
        std::cout << item.first << ":  " << item.second << std::endl;
    }
    std::cout << std::endl;
}

void mAP::printMatches(const std::vector<std::pair<int, std::string>>& matches) {
    std::cout << "ID:  Match" << std::endl;
    for (const auto& item : matches) {
        std::cout << item.first << ":  " << item.second << std::endl;
    }
    std::cout << std::endl;
}

void mAP::printCumulativeTP(const std::vector<std::pair<int, int>>& cumulativeTP) {
    std::cout << "ID:  Cumulative TP" << std::endl;
    for (const auto& tpItem : cumulativeTP) {
        std::cout << tpItem.first << ":  " << tpItem.second << std::endl;
    }
    std::cout << std::endl;
}

void mAP::printCumulativeFP(const std::vector<std::pair<int, int>>& cumulativeFP) {
    std::cout << "ID:  Cumulative FP" << std::endl;
    for (const auto& fpItem : cumulativeFP) {
        std::cout << fpItem.first << ":  " << fpItem.second << std::endl;
    }
    std::cout << std::endl;
}

void mAP::printPrecision(const std::vector<std::pair<int, float>>& precisionValues) {
    std::cout << "ID:  Precision" << std::endl;
    for (size_t i = 0; i < precisionValues.size(); ++i) {
        std::cout << precisionValues[i].first << ":  " << precisionValues[i].second << std::endl;
    }
    std::cout << std::endl;
}

void mAP::printRecall(const std::vector<std::pair<int, float>>& recallValues) {
    std::cout << "ID:  Recall" << std::endl;
    for (const std::pair<int, float>& item : recallValues) {
        std::cout << item.first << ":  " << item.second << std::endl;
    }
    std::cout << std::endl;
}

void mAP::printIoUValues(const std::vector<std::pair<int, float>>& iouValues) {
    std::cout << "ID:  IoU" << std::endl;
    for (const auto& iouItem : iouValues) {
        std::cout << iouItem.first << ":  " << iouItem.second << std::endl;
    }
    std::cout << std::endl;
}

std::vector<std::pair<int, float>> mAP::computeIoU(const std::map<int, std::vector<int>>& gtbb, const std::map<int, std::vector<int>>& pred) {
    if (gtbb.size() != pred.size()) {
        std::cout << "Exception: The maps' size is different!" << std::endl;
    }

    // Computing IoU
    std::map<int, std::vector<int>> intersections = computeIntersection(gtbb, pred);

    std::vector<int> areaGT = getArea(gtbb);
    std::vector<int> areaPred = getArea(pred);
    std::vector<int> areaIntersections = getArea(intersections);
    std::vector<float> areaUnion;
    std::vector<float> iou;

    for (size_t i = 0; i < areaGT.size(); i++) {
        areaUnion.push_back(areaGT[i] + areaPred[i] - areaIntersections[i]);
        iou.push_back(areaIntersections[i] / areaUnion[i]);
    }

    // Inserting Id and its relative IoU in a map
    std::map<int, float> pred_iou;

    // Putting the text in the bottom left corner describing the IoU for each intersection
    size_t i = 0;
    for (const auto& item : intersections) {
        pred_iou.insert({ item.first, iou[i] });
        i++;
    }

    // Declare a vector of pairs
    std::vector<std::pair<int, float>> pairVec;
    for (const auto& item : pred_iou) {
        pairVec.push_back(item);
    }

    // Sort in descending order the key-value pairs (id - IoU).
    // For this purpose, we use a vector of pairs instead of a map
    std::sort(pairVec.begin(), pairVec.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
        });

    return pairVec;
}

std::vector<std::pair<int, std::string>> mAP::classifyMatches(const std::vector<std::pair<int, float>>& iouData) {
    
    std::vector<std::pair<int, std::string>> matchVector;

    for (const auto& data : iouData) {
        std::string match;
        if (data.second == 0.0) {
            match = "FN";
        }
        else if (data.second <= IOU_THRESHOLD) {
            match = "FP";
        }
        else {
            match = "TP";
        }
        matchVector.push_back(std::make_pair(data.first, match));
    }
    return matchVector;
}

std::vector<std::pair<int, int>> mAP::calculateCumulativeTP(const std::vector<std::pair<int, std::string>>& predictions) {
    std::vector<std::pair<int, int>> cumulativeTP;
    int cumTP = 0;

    for (const auto& prediction : predictions) {
        if (prediction.second == "TP") {
            cumTP += 1;
        }
        cumulativeTP.push_back(std::make_pair(prediction.first, cumTP));
    }

    return cumulativeTP;
}

std::vector<std::pair<int, int>> mAP::calculateCumulativeFP(const std::vector<std::pair<int, std::string>>& predictions) {
    std::vector<std::pair<int, int>> cumulativeFP;
    int cumFP = 0;

    for (const auto& prediction : predictions) {
        if (prediction.second == "FP") {
            cumFP += 1;
        }
        cumulativeFP.push_back(std::make_pair(prediction.first, cumFP));
    }

    return cumulativeFP;
}

std::vector<std::pair<int, float>> mAP::calculatePrecision(const std::vector<std::pair<int, int>>& truePositives, const std::vector<std::pair<int, int>>& falsePositives) {
    std::vector<std::pair<int, float>> precision;
    std::vector<int> truePositivesVec;

    for (const auto& tpItem : truePositives) {
        truePositivesVec.push_back(tpItem.second);
    }

    int i = 0;
    for (const auto& fpItem : falsePositives) {
        float prec = truePositivesVec[i] / static_cast<float>(truePositivesVec[i] + fpItem.second);
        precision.push_back(std::make_pair(fpItem.first, prec));
        i++;
    }
    return precision;
}

std::vector<std::pair<int, float>> mAP::calculateRecall(const std::vector<std::pair<int, int>>& cumulativeTruePositives, float totalGroundTruths) {
    std::vector<std::pair<int, float>> recall;

    for (const auto& tpItem : cumulativeTruePositives) {
        float rec = tpItem.second / totalGroundTruths;
        recall.push_back(std::make_pair(tpItem.first, rec));
    }

    return recall;
}


std::vector<std::pair<int, float>> mAP::interpolatePrecision(const std::vector<std::pair<int, float>>& precisionValues, const std::vector<std::pair<int, float>>& recallValues) {
    // Sort the precision and recall vectors in ascending order based on the intensity (first element of the pair).
    std::vector<std::pair<int, float>> sortedPrecisionValues = precisionValues;
    std::sort(sortedPrecisionValues.begin(), sortedPrecisionValues.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
        });

    std::vector<std::pair<int, float>> sortedRecallValues = recallValues;
    std::sort(sortedRecallValues.begin(), sortedRecallValues.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
        });

    // Create vectors to store the interpolated precision and recall values.
    std::vector<float> interpolatedPrecision;
    std::vector<float> interpolatedRecall;

    // Loop over the recall values and interpolate precision at each recall point.
    float maxPrecision = 0.0;
    for (const auto& recall : sortedRecallValues) {
        float recallVal = recall.second;
        float precisionVal = 0.0;

        // Find the precision value corresponding to the closest recall value in the precisionValues vector.
        for (const auto& precision : sortedPrecisionValues) {
            if (precision.first >= recall.first) {
                precisionVal = precision.second;
                break;
            }
        }

        // Update the maximum precision value.
        if (precisionVal > maxPrecision) {
            maxPrecision = precisionVal;
        }

        // Store the interpolated precision and recall values.
        interpolatedPrecision.push_back(maxPrecision);
        interpolatedRecall.push_back(recallVal);
    }

    // Calculate the average precision using the trapezoidal rule for numerical integration.
    std::vector<std::pair<int, float>> averagePrecisionPairs;
    float previousRecall = 0.0;
    float accumulatedPrecision = 0.0;

    for (size_t i = 0; i < interpolatedRecall.size(); i++) {
        float recallDiff = interpolatedRecall[i] - previousRecall;
        accumulatedPrecision += interpolatedPrecision[i] * recallDiff;
        previousRecall = interpolatedRecall[i];

        // Store the intensity and the average precision at this point
        averagePrecisionPairs.push_back(std::make_pair(sortedRecallValues[i].first, accumulatedPrecision));
    }

    return averagePrecisionPairs;
}

void mAP::printInterpolatedPrecision(std::vector<std::pair<int, float>> interpolatedPrecision) {
    std::cout << "Interpolation " << std::endl;

    for (const auto& pair : interpolatedPrecision) {
        std::cout << "ID: " << pair.first << ", Interpolated Precision: " << pair.second << std::endl;
    }
}
