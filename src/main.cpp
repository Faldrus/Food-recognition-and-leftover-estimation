#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "detectDishes.h"
#include "utils.h"

using namespace cv;
using namespace std;


int main()
{
    const vector<string> label = getLabels();

    vector<Mat> tray1, tray2, tray3, tray4, tray5, tray6, tray7, tray8;
    tray1 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray1/");
    tray2 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray2/");
    tray3 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray3/");
    tray4 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray4/");
    tray5 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray5/");
    tray6 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray6/");
    tray7 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray7/");
    tray8 = createVecImgFromSource("../src/resource/Food_leftover_dataset/tray8/");
    vector<vector<Mat>> trays = {tray1,tray2,tray3,tray4,tray5,tray6,tray7,tray8};



    //vectors with dish from a trayN
    vector<Mat> firstTray1,firstTray2,firstTray3,firstTray4,firstTray5,firstTray6,firstTray7,firstTray8;
    vector<Mat> secondTray1,secondTray2,secondTray3,secondTray4,secondTray5,secondTray6,secondTray7,secondTray8;
    vector<Mat> saladTray1,saladTray2,saladTray3,saladTray4,saladTray5,saladTray6,saladTray7,saladTray8;
    vector<Mat> breadTray1,breadTray3,breadTray4,breadTray5,breadTray6,breadTray7,breadTray8;

    //vectors used to select the correct dish for each trayN
    vector<Mat> temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8;


    temp1 = getFoodImageByAllTrays(trays,0);
    temp2 = getFoodImageByAllTrays(trays,1);
    temp3 = getFoodImageByAllTrays(trays,2);
    temp4 = getFoodImageByAllTrays(trays,3);
    temp5 = getFoodImageByAllTrays(trays,4);
    temp6 = getFoodImageByAllTrays(trays,5);
    temp7 = getFoodImageByAllTrays(trays,6);
    temp8 = getFoodImageByAllTrays(trays,7);

    //SALAD + BREAD
    for(int i=0;i<4;i++)
    {
        saladTray1.push_back(detectSalad(tray1[i]));
        saladTray2.push_back(detectSalad(tray2[i]));
        saladTray3.push_back(detectSalad(tray3[i]));
        saladTray4.push_back(detectSalad(tray4[i]));
        saladTray5.push_back(detectSalad(tray5[i]));
        saladTray6.push_back(detectSalad(tray6[i]));
        saladTray7.push_back(detectSalad(tray7[i]));
        saladTray8.push_back(detectSalad(tray8[i]));

        breadTray1.push_back(detectBreadByHisto(tray1[i],1));
        breadTray4.push_back(detectBreadByHisto(tray4[i],4));
        breadTray5.push_back(detectBreadByHisto(tray5[i],5));
    }

    // TRAY1
    firstTray1.push_back(temp1[1]);
    firstTray1.push_back(temp1[3]);
    firstTray1.push_back(temp1[4]);
    firstTray1.push_back(temp1[6]);
    secondTray1.push_back(temp1[0]);
    secondTray1.push_back(temp1[2]);
    secondTray1.push_back(temp1[5]);
    secondTray1.push_back(temp1[7]);

    //TRAY2
    firstTray2.push_back(temp2[0]);
    firstTray2.push_back(temp2[2]);
    firstTray2.push_back(temp2[5]);
    firstTray2.push_back(temp2[6]);
    secondTray2.push_back(temp2[1]);
    secondTray2.push_back(temp2[3]);
    secondTray2.push_back(temp2[4]);
    secondTray2.push_back(temp2[7]);

    //TRAY3
    firstTray3.push_back(temp3[0]);
    firstTray3.push_back(temp3[2]);
    firstTray3.push_back(temp3[5]);
    firstTray3.push_back(temp3[6]);
    secondTray3.push_back(temp3[1]);
    secondTray3.push_back(temp3[3]);
    secondTray3.push_back(temp3[4]);
    secondTray3.push_back(temp3[7]);

    //TRAY4
    firstTray4.push_back(temp4[0]);
    firstTray4.push_back(temp4[2]);
    firstTray4.push_back(temp4[5]);
    firstTray4.push_back(temp4[6]);
    secondTray4.push_back(temp4[1]);
    secondTray4.push_back(temp4[3]);
    secondTray4.push_back(temp4[4]);
    secondTray4.push_back(temp4[7]);

    //TRAY5
    firstTray5.push_back(temp5[0]);
    firstTray5.push_back(temp5[2]);
    firstTray5.push_back(temp5[5]);
    firstTray5.push_back(temp5[1]);
    secondTray5.push_back(temp5[3]);
    secondTray5.push_back(temp5[4]);
    secondTray5.push_back(temp5[6]);

    //TRAY6
    firstTray6.push_back(temp6[0]);
    firstTray6.push_back(temp6[3]);
    firstTray6.push_back(temp6[4]);
    firstTray6.push_back(temp6[7]);
    secondTray6.push_back(temp6[1]);
    secondTray6.push_back(temp6[2]);
    secondTray6.push_back(temp6[5]);
    secondTray6.push_back(temp6[6]);

    //TRAY7
    firstTray7.push_back(temp7[0]);
    firstTray7.push_back(temp7[2]);
    firstTray7.push_back(temp7[5]);
    firstTray7.push_back(temp7[7]);
    secondTray7.push_back(temp7[1]);
    secondTray7.push_back(temp7[3]);
    secondTray7.push_back(temp7[4]);
    secondTray7.push_back(temp7[6]);

    //TRAY8
    firstTray8.push_back(temp8[0]);
    firstTray8.push_back(temp8[3]);
    firstTray8.push_back(temp8[4]);
    firstTray8.push_back(temp8[7]);
    secondTray8.push_back(temp8[1]);
    secondTray8.push_back(temp8[2]);
    secondTray8.push_back(temp8[5]);
    secondTray8.push_back(temp8[6]);

    //vector with only the box images
    vector<Mat>boxFirst1,boxFirst2,boxFirst3,boxFirst4,boxFirst5,boxFirst6,boxFirst7,boxFirst8;
    vector<Mat>boxSecond1,boxSecond2,boxSecond3,boxSecond4,boxSecond5,boxSecond6,boxSecond7,boxSecond8;
    vector<Mat>boxSalad1,boxSalad2,boxSalad3,boxSalad4,boxSalad5,boxSalad6,boxSalad7,boxSalad8;
    vector<Mat>boxBread1,boxBread2,boxBread3,boxBread4,boxBread5,boxBread6,boxBread7,boxBread8;


    //Vector with the Rect for each image, use to obtain the values of the Rect
    vector<Rect>boxFirstRect1,boxFirstRect2,boxFirstRect3,boxFirstRect4,boxFirstRect5,boxFirstRect6,boxFirstRect7,boxFirstRect8;
    vector<Rect>boxSecondRect1,boxSecondRect2,boxSecondRect3,boxSecondRect4,boxSecondRect5,boxSecondRect6,boxSecondRect7,boxSecondRect8;
    vector<Rect>boxSaladRect1,boxSaladRect2,boxSaladRect3,boxSaladRect4,boxSaladRect5,boxSaladRect6,boxSaladRect7,boxSaladRect8;
    vector<Rect>boxBreadRect1,boxBreadRect2,boxBreadRect3,boxBreadRect4,boxBreadRect5,boxBreadRect6,boxBreadRect7,boxBreadRect8;

    //Vector with output images
    vector<Mat> output1, output2,output3,output4,output5,output6,output7,output8;

    //Need to call viewBoxOnImage to obtain the vector with only the box images. Same for the values of Rect
    //It needs the vector for each type of food, the original image,an int with the number of tray, boxes for each type of food and the output vector
    output7 = viewBoxOnImage(firstTray7,saladTray7,secondTray7,breadTray7,tray7,output7,boxFirst7,boxSecond7,boxSalad7,boxBread7,boxFirstRect7,boxSecondRect7,boxSaladRect7,boxBreadRect7,7);

    for(int i=0;i<4;i++)
    {
        imshow("output",output7[i]);
        waitKey(0);
    }

    return 0;
}