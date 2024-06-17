cmake_minimum_required(VERSION 3.25)
project(Food_recognition_and_leftover_estimation)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Food_recognition_and_leftover_estimation
        src/detectDishes.h src/detectDishes.cpp src/utils.h src/utils.cpp src/main.cpp src/Performance/FoodLeftoverEstimation.cpp src/Performance/mAP.cpp src/Performance/mIOU.cpp src/Detector/FoodTemplate.cpp src/Detector/ObjectRecognizer.cpp)

target_link_libraries(Food_recognition_and_leftover_estimation PRIVATE ${OpenCV_LIBS})
