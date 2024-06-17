#include "detectDishes.h"
#include "utils.h"

using namespace std;
using namespace cv;


//Function that given an image, returns a new image with only the dishes in the picture
Mat detectDishesEdge(const Mat& image)
{
    int max= 350;
    int min = 250;
    int hCanny = 100;
    int hCircle = 50;
    Mat imageCircles;
    image.copyTo(imageCircles);
    Mat gray;
    vector<Vec3f>circles;

    cvtColor(image,gray,COLOR_RGB2GRAY);

    medianBlur(gray,gray,7);

    Mat mask(imageCircles.size(), CV_8UC1, Scalar(0));
    //HoughCircles(gray,circles,HOUGH_GRADIENT,1,gray.rows/16,100,30,268,292);
    // HoughCircles(gray, circles, HOUGH_GRADIENT, 1,220,100, 20, 260, 280);
    HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, min, max);
    for (int i =0; i<circles.size();i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0],c[1]);
        int radius = c[2];
        circle(mask, center, radius, Scalar(255), -1);
    }
    for(int i=0;i<imageCircles.rows;i++)
    {
        for(int j=0;j<imageCircles.cols;j++)
        {
            if(mask.at<unsigned char>(i,j)==0)
            {
                imageCircles.at<Vec3b>(i,j) = 0;
            }
        }
    }
    return imageCircles;
}

//Function that given an image, returns a new image with only the salad in the picture
Mat detectSalad(const Mat& image)
{
    int hCanny = 100;
    int hCircle = 50;
    int max = 210;
    int min = 175;
    Mat salad;
    Mat gray;
    vector<Vec3f> bowl;
    image.copyTo(salad);
    Mat noSalad(salad.size(), salad.type(),Scalar(0, 0, 0));

    cvtColor(image,gray,COLOR_RGB2GRAY);

    HoughCircles(gray, bowl, HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, min, max);
    for (int i = 0; i < bowl.size(); i++)
        circle(salad, Point(bowl[i][0], bowl[i][1]), bowl[i][2], Scalar(0, 0, 0), 2);

    if(bowl.empty())
    {
        cout<<"No salad in that image"<<endl;
        return noSalad;
    }
    //cout<<"bowl size:"<<bowl.size()<<endl;
    Mat mask(image.size(), CV_8UC1,Scalar(0));
    //cout<<"bowlSize"<<bowl.size()<<endl;

    Vec3i c = bowl[0];
    Point center = Point(c[0],c[1]);
    int radius = c[2];
    circle(mask, center, radius, Scalar(255), -1);

    for(int i=0;i<salad.rows;i++)
    {
        for(int j=0;j<salad.cols;j++)
        {
            if(mask.at<unsigned char>(i,j)==0)
            {
                salad.at<Vec3b>(i,j) = 0;
            }
        }
    }

    return salad;

}

//Function that given an image, returns a new image with dishes and salad in the picture
Mat detectFoods(const Mat& image,int n)
{
    vector<Mat>images;
    Mat temp1, temp2;
    if(n==4)
        temp1 = detectDishesEdge4(image);
    else
        temp1 = detectDishesEdge(image);
    temp2 = detectSalad(image);
    Mat all = temp1 + temp2;
    return all;
}


//Function that given an image, returns a new image with only the bread in the picture
// not used
Mat detectBread(const Mat& image)
{
    const int THRESHOLD = 32;
    // Convert to HSV color space
    Mat hsv_image;
    cvtColor(image, hsv_image, COLOR_RGB2HSV);
    // Threshold the HSV image

    Mat thresholded_HSV;
    extractChannel(hsv_image, thresholded_HSV, 1);
    normalize(thresholded_HSV, thresholded_HSV, 0, 255, NORM_MINMAX);
    threshold(thresholded_HSV, thresholded_HSV, THRESHOLD, 255, THRESH_BINARY);

    // Filter the areas (to remove small objects and get only food blobs) DA RIVEDERE
    vector<vector<Point>> contours;
    findContours(thresholded_HSV, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for(int i=0; i<contours.size(); i++)
    {
        if(contourArea(contours[i]) > 4000)
        {
            drawContours(thresholded_HSV, contours, i, 255, -1);
        }
    }

    // Apply morphological operations to improve the mask
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(thresholded_HSV, thresholded_HSV, MORPH_CLOSE, kernel);
    //morphologyEx(thresholded_HSV, thresholded_HSV, MORPH_OPEN, kernel);

    // Apply max filter to remove small noise
    Mat max_filtered;
    dilate(thresholded_HSV, max_filtered, kernel);

    return max_filtered;
}

//Function that finds bread based on histogram and morphological operations and colors
Mat detectBreadByHisto(const Mat& image,int n)
{
    Mat external = preparePhoto(image,n);
    Mat photo;
    image.copyTo(photo);
    //Mat external, temp;
    //temp = detectFoods(image);
    //external = image - temp;
    Mat bread;
    int size = 5;
    int numColors = 5;
    int delta = 40;
    int thresold = 60;

    Scalar targetColor1(38, 187, 181);
    Scalar targetColor2(2,53,73);
    Scalar targetColor3(3,118,121);
    Scalar targetColor4(84,124,153);
    Scalar targetColor5(21,165,160);
    Scalar targetColor6(255,255,255);
    Scalar targetColor7(2,215,206);
    Scalar targetColor8(40,92,186);
    Scalar targetColor9(231,208,192);
    Scalar targetColor10(31,72,171);
    Scalar targetColor11(61,217,219);
    Scalar targetColor12(107,204,198);
    Scalar targetColor13(79,206,203);
    Scalar targetColor14(14,46,131);
    Scalar targetColor15(8,31,118);
    Scalar targetColor16(31,3,0);
    Scalar targetColor17(212,135,67);

    removeSimilarPixels(external,targetColor1,thresold);
    removeSimilarPixels(external,targetColor2,thresold);
    removeSimilarPixels(external,targetColor3,thresold);
    removeSimilarPixels(external,targetColor4,thresold);
    removeSimilarPixels(external,targetColor5,thresold);
    removeSimilarPixels(external,targetColor6,thresold);
    removeSimilarPixels(external,targetColor7,thresold);
    removeSimilarPixels(external,targetColor8,thresold);
    removeSimilarPixels(external,targetColor9,thresold);
    removeSimilarPixels(external,targetColor10,thresold);
    removeSimilarPixels(external,targetColor12,thresold);
    removeSimilarPixels(external,targetColor13,thresold);
    removeSimilarPixels(external,targetColor14,thresold);
    removeSimilarPixels(external,targetColor15,thresold);
    removeSimilarPixels(external,targetColor16,thresold);
    removeSimilarPixels(external,targetColor17,thresold);


    //rimuovo un bel po' di cose ma anche troppe
    bread = removeColors(external,size,numColors,delta);
    bread = removeDishes(bread,20);

    Mat grayImage;
    Mat mask;

    cvtColor(bread, grayImage, COLOR_BGR2GRAY);
    mask = Mat::zeros(grayImage.size(),grayImage.type());
    grayImage.copyTo(bread);


    for(int i=0;i<image.rows;i++)
    {
        for(int j = 0;j<image.cols;j++)
        {
            if(grayImage.at<unsigned char>(i,j) != 0)
                mask.at<unsigned char>(i,j) = 255;
        }
    }


    //return mask;

    //Use 2 different kernel, one for erosion and one for dilation
    Mat erodeKernel, dilateKernel;

    erodeKernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    dilateKernel = getStructuringElement(MORPH_RECT, Size(16, 16));
   
    //Apply erosion on mask
    erode(mask, mask, erodeKernel);
    
    //Apply dilation on mask
    dilate(mask, mask, dilateKernel);

    Mat binaryImage;
    threshold(mask, binaryImage, 127, 255, THRESH_BINARY);

    // Trovare i contorni nell'immagine binaria
    vector<vector<Point>> contours;
    findContours(binaryImage, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Trovare il contorno con l'area maggiore
    double maxArea = 0;
    int maxAreaIndex = -1;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area > maxArea)
        {
            maxArea = area;
            maxAreaIndex = i;
        }
    }

    // Crea una maschera dell'area di interesse (again)
    Mat mask1 = Mat::zeros(mask.size(), CV_8UC1);
    if (maxAreaIndex >= 0)
    {
        drawContours(mask1, contours, maxAreaIndex, Scalar(255), FILLED);
    }


    // Applicare la maschera all'immagine originale
    Mat outputImage;
    image.copyTo(outputImage, mask1);

    //I need to specify the cv::
    Rect boundingRect = cv::boundingRect(contours[maxAreaIndex]);

    outputImage = removeDishes(outputImage,30);
    outputImage= removeColors(outputImage,15,15,20);

    cvtColor(outputImage,outputImage,COLOR_BGR2GRAY);
    Mat kernelss = getStructuringElement(MORPH_RECT, Size(71, 71));
    Mat closedImage;
    morphologyEx(outputImage, closedImage, MORPH_CLOSE, kernelss);

    for(int i =0;i<closedImage.rows;i++)
    {
        for(int j =0;j<closedImage.cols;j++)
        if(closedImage.at<unsigned char>(i,j)==0)
        {
            photo.at<Vec3b>(i,j)[0]=0;
            photo.at<Vec3b>(i,j)[1]=0;
            photo.at<Vec3b>(i,j)[2]=0;
        }
    }

   // photo = drawBox(photo,Scalar(255,255,0));

    return photo;
}

//Function that removes background elements around the desired object
Mat removeDishes(Mat image, int delta)
{
    Mat img;
    image.copyTo(img);
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
            Vec3b pix = img.at<Vec3b>(i,j);
            int avg = (int)(pix[0] + pix[1] + pix[2])/3;
            if(pix[0]-avg < delta && pix[1]-avg < delta && pix[2]-avg < delta)
            {
                img.at<Vec3b>(i,j) [0] = 0;
                img.at<Vec3b>(i,j) [1] = 0;
                img.at<Vec3b>(i,j) [2] = 0;
            }
        }
    }
    return img;
}

//Function that gets the yogurt mask to remove it
Mat getMaskYogurt(const Mat& img)
{
    Mat image;
    img.copyTo(image);
    Mat hsvImage;
    Mat result;
    Mat out;
    Mat grayImage;
    Mat mask;
    Scalar lowerBlue = Scalar(90, 50, 50);
    Scalar upperBlue = Scalar(130, 255, 255);

    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    inRange(hsvImage, lowerBlue, upperBlue, out);
    image.copyTo(result, out);
    result.copyTo(image);

    cvtColor(image, grayImage, COLOR_HSV2BGR);
    cvtColor(grayImage, grayImage, COLOR_BGR2GRAY);
    mask = Mat::zeros(grayImage.size(),grayImage.type());
    grayImage.copyTo(image);
    for(int i=0;i<image.rows;i++)
    {
        for(int j = 0;j<image.cols;j++)
        {
            if(grayImage.at<unsigned char>(i,j) != 0)
                mask.at<unsigned char>(i,j) = 255;
        }
    }

    Mat kernel = getStructuringElement(MORPH_RECT, Size(33, 33));
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);
    return mask;
}

//Function that removes dishes, yogurt, salad from the image
Mat preparePhoto(const Mat& img,int n)
{
    Mat image;
    img.copyTo(image);
    Mat temp = detectFoods(image,n);
    Mat external = image - temp;
    Mat yogurtMask = getMaskYogurt(external);
    for(int i=0;i<external.rows;i++)
    {
        for(int j=0;j<external.cols;j++)
        {
            if(yogurtMask.at<unsigned char>(i,j)==255)
            {
                external.at<Vec3b>(i,j)[0] = 0;
                external.at<Vec3b>(i,j)[1] = 0;
                external.at<Vec3b>(i,j)[2] = 0;
            }
        }
    }
    return external;
}

//creo immagine con tutto e sfondo nero
//Mat allF(const Mat& img)
//{
//    Mat image;
//    img.copyTo(image);
//    Mat external = preparePhoto(image);
//    Mat bread = detectBreadByHisto(external);
//    Mat temp = detectFoods(image);
//    Mat all = bread + temp;
//    return  all;
//}

//Function that returns a vector containing individual dish from an image with multiple plates
vector<Mat> getOneDish(const Mat &image)
{
    Mat gray;
    Mat image1 = image.clone();
    cvtColor(image, gray, COLOR_RGB2GRAY);
    medianBlur(gray, gray, 7);
    //tray4 383,230
    //base 350,250
    int maxRadius = 350;
    int minRadius = 250;
    int hCanny = 100;
    int hCircle = 50;


    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, minRadius, maxRadius);

    vector<Mat> dishImages;
    for (const Vec3f& c : circles)
    {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        Mat mask(image.size(), CV_8UC1, Scalar(0));
        circle(mask, center, radius, Scalar(255), -1);
        Mat dishImage;
        image.copyTo(dishImage, mask);
        dishImages.push_back(dishImage);
    }

    return dishImages;
}

//Function to segment the first dish
Mat segmentFirst(Mat image)
{
    Mat pastasugo = image.clone();

    Mat bgrImage;
    cvtColor(image, bgrImage, COLOR_RGBA2BGR);

    Mat mask(bgrImage.size(), CV_8UC1, GC_PR_BGD);
    Rect rectangle(50, 50, bgrImage.cols - 100, bgrImage.rows - 100);
    grabCut(bgrImage, mask, rectangle, Mat(), Mat(), 5, GC_INIT_WITH_RECT);
    Mat foregroundMask = (mask ==GC_PR_FGD) | (mask == GC_FGD);

    //è una maschera del piatto
    Mat binaryMask;
    foregroundMask.convertTo(binaryMask, CV_8U, 255);


    //"separo" con colori piatto e cibo
    Mat pastaImage;
    bgrImage.copyTo(pastaImage, binaryMask);


    //fa una prima segmentazione, già efficace solo nella foto con tutta la pasta (non mangiata)
    pastaImage = removeDishes(pastaImage,15);


    //sfrutto funzione usata per rimovere cose attorno al pane semplificata
    Mat pasta;
    int numColors = 105;
    int delta = 40;
    pastaImage = removeColors(image,numColors,numColors,delta);




    Mat grayPasta;
    cvtColor(pastaImage, grayPasta, COLOR_BGR2GRAY);

    //maschera di quello trovato fino a qua
    int thresholdValue = 1;
    Mat pastaMask;
    threshold(grayPasta, pastaMask, thresholdValue, 255, THRESH_BINARY);


    //maschera più precisa anche se forse non cambia niente, da rivedere
    vector<vector<Point>> contours;
    findContours(pastaMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat finalMask = Mat::zeros(pastaMask.size(), CV_8UC1);
    drawContours(finalMask, contours, -1, Scalar(255), FILLED);
    Mat pastaWithMask;
    pastaImage.copyTo(pastaWithMask, finalMask);
    pastaWithMask = removeDishes(pastaWithMask,19);


    //agisce sull'immagine a colori
    for(int i=0;i<image.rows;i++)
    {
        for(int j =0;j<image.cols;j++)
        {
            if(pastaWithMask.at<Vec3b>(i,j)==Vec3b(0,0,0))
                pastasugo.at<Vec3b>(i,j) = Vec3b (0,0,0);
        }
    }


    //toglie un po di schifi e sughetto NB mi buca la prima
    Mat out_gray = pastasugo.clone();
    cvtColor(out_gray, out_gray, COLOR_BGR2GRAY);
    Mat kernelone = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(out_gray, out_gray, MORPH_OPEN, kernelone);


    //cvtColor(out_gray, out_gray, COLOR_BGR2GRAY);
    vector<vector<Point>> contours1;
    findContours(out_gray, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Crea una maschera bianca con l'interno del contorno riempito di bianco
    Mat maskWithContours = Mat::zeros(out_gray.size(), CV_8UC1);
    drawContours(maskWithContours, contours1, -1, Scalar(255), FILLED);


    //agisce sulla foto a colori
    for(int i=0;i<image.rows;i++)
    {
        for(int j =0;j<image.cols;j++)
        {
            if(maskWithContours.at<unsigned char>(i,j)==0)
                pastasugo.at<Vec3b>(i,j) = Vec3b (0,0,0);
        }
    }

    //cerco di tappare i buchi
    Mat kerneltwo = getStructuringElement(MORPH_RECT, Size(133, 133));
    morphologyEx(maskWithContours, maskWithContours, MORPH_CLOSE, kerneltwo);

    return maskWithContours;
}

//Function that calls segmentFirst and sets the pixels outside the mask to zero.
//Used to perform the final form of segmentation for first dish
Mat getFirst(Mat image)
{
    Mat temp = image.clone();
    Mat out = image.clone();
    Mat first = segmentFirst(temp);
    for(int i=0;i<temp.rows;i++)
    {
        for(int j =0;j<temp.cols;j++)
        {
            if(first.at<unsigned char>(i,j)==0)
            {
                temp.at<Vec3b>(i,j)[0] =0;
                temp.at<Vec3b>(i,j)[1] =0;
                temp.at<Vec3b>(i,j)[2] =0;
            }
        }
    }
    //25 ok
    temp = removeDishes(temp,25);
    //return temp;
    temp = removeColors(temp,10,10,70);


    cvtColor(temp, temp, COLOR_BGR2GRAY);
    vector<vector<Point>> contours1;
    findContours(temp, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Crea una maschera bianca con l'interno del contorno riempito di bianco
    Mat maskWithContours = Mat::zeros(temp.size(), CV_8UC1);
    drawContours(maskWithContours, contours1, -1, Scalar(255), FILLED);


    Mat kerneltwo = getStructuringElement(MORPH_RECT, Size(52, 52));
    morphologyEx(maskWithContours, maskWithContours, MORPH_CLOSE, kerneltwo);


    for(int i=0;i<temp.rows;i++)
    {
        for(int j =0;j<temp.cols;j++)
        {
            if(maskWithContours.at<unsigned char>(i,j)==0)
            {
                out.at<Vec3b>(i,j)[0] =0;
                out.at<Vec3b>(i,j)[1] =0;
                out.at<Vec3b>(i,j)[2] =0;
            }
        }
    }


    //
    //out = drawBox(out,Scalar(0,0,255));

    return out;

}

//Function that returns a vector containing the individual dishes in each tray
//set value of i and j to work on specific trays
vector<Mat> getFoodImageByAllTrays(vector<vector<Mat>> trays,int nTray)
{
    vector<Mat> dishesOfAllTrays;
    vector<Mat> all;
    for(int i =0;i<4;i++)
    {
        //for(int j=0;j<1;j++)
        //{
            Mat dishes = trays[nTray][i];
            //cout<<"i:" << i<< "j:"<<endl;
            if(nTray==3)
                all = getOneDish4(dishes);
            else
                all = getOneDish(dishes);

            if(all.size()==1)
            {
                Mat dish1;
                dish1 = all[0];
                dishesOfAllTrays.push_back(dish1);
                continue;
            }

            Mat dish1, dish2;
            dish1 = all[0];
            dish2 = all[1];
            dishesOfAllTrays.push_back(dish1);
            dishesOfAllTrays.push_back(dish2);
        //}
    }
    return  dishesOfAllTrays;
}

//Function that segment the salad
Mat segmentSalad(Mat img)
{
    Mat salad = img.clone();//detectSalad(img.clone());
    Mat filter;
    bilateralFilter(salad, filter, 15, 1000*0.1, 800*0.01);

    Mat saladHSV;
    cvtColor(salad, saladHSV, COLOR_BGR2HSV);

    vector<Mat> hsvChan;
    split(saladHSV, hsvChan);
    Mat satChannel = hsvChan[1];
    Mat mask;
    int lower = 150;
    int upper = 255;
    inRange(satChannel, lower, upper, mask);

    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    for (int i = 0; i < 20; i++)
    {
        morphologyEx(mask, mask, MORPH_CLOSE, element);
    }

    floodFill(mask, Point(0,0), Scalar(125));
    inRange(mask, 123, 125, mask);

    int count;
    int size = 15;
    int area = size*size;
    Mat outMask;
    Mat labels;
    Mat stats;
    Mat centroids;
    Mat seg;

    bitwise_not(mask,outMask);
    count = connectedComponentsWithStats(outMask, labels, stats, centroids);
    for(int i = 0; i<count; i++)
    {
        int compSize = stats.at<int>(i, CC_STAT_AREA);
        if(compSize < area)
        {
            Mat temp = (labels == i);
            outMask.setTo(0,temp);
        }
    }

    salad.copyTo(seg, outMask);

   // seg = drawBox(seg,Scalar(0,255,0));
    return seg; //return seg
}

//Function to set the radius for tray4 (works as detectDishesEdges)
Mat detectDishesEdge4(const Mat& image)
{
    int max= 383;
    int min = 230;
    int hCanny = 100;
    int hCircle = 50;
    Mat imageCircles;
    image.copyTo(imageCircles);
    Mat gray;
    vector<Vec3f>circles;

    cvtColor(image,gray,COLOR_RGB2GRAY);

    medianBlur(gray,gray,7);

    Mat mask(imageCircles.size(), CV_8UC1, Scalar(0));
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, min, max);
    for (int i =0; i<circles.size();i++)
    {
        Vec3i c = circles[i];
        Point center = Point(c[0],c[1]);
        int radius = c[2];
        circle(mask, center, radius, Scalar(255), -1);
    }

    for(int i=0;i<imageCircles.rows;i++)
    {
        for(int j=0;j<imageCircles.cols;j++)
        {
            if(mask.at<unsigned char>(i,j)==0)
            {
                imageCircles.at<Vec3b>(i,j) = 0;
            }
        }
    }
    return imageCircles;
}

//Function to set the radius for tray4 (works as detectOneDish)
vector<Mat> getOneDish4(const Mat &image)
{
    Mat gray;
    cvtColor(image, gray, COLOR_RGB2GRAY);
    medianBlur(gray, gray, 7);
    //tray4 383,230
    //base 350,250
    int maxRadius = 383; int minRadius = 230; int hCanny = 100; int hCircle = 50;

    vector<cv::Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 16, hCanny, hCircle, minRadius, maxRadius);

    vector<Mat> dishImages;
    for (const Vec3f& c : circles)
    {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        Mat mask(image.size(), CV_8UC1, Scalar(0));
        circle(mask, center, radius, Scalar(255), -1);
        Mat dishImage;
        image.copyTo(dishImage, mask);
        dishImages.push_back(dishImage);
    }

    return dishImages;
}

//Function to segment the second dish
Mat segmentSecond(const Mat& image){
    Mat secondo = image.clone();

    Mat bgrImage;
    cvtColor(image, bgrImage, COLOR_RGBA2BGR);

    Mat mask(bgrImage.size(), CV_8UC1, GC_PR_BGD);
    Rect rectangle(100, 100, bgrImage.cols - 100, bgrImage.rows - 100);
    grabCut(bgrImage, mask, rectangle, Mat(), Mat(), 5, GC_INIT_WITH_RECT);
    Mat foregroundMask = (mask ==GC_PR_FGD) | (mask == GC_FGD);

    //è una maschera del piatto
    Mat binaryMask;
    foregroundMask.convertTo(binaryMask, CV_8U, 255);

    //"separo" con colori piatto e cibo
    Mat secondImage;
    bgrImage.copyTo(secondImage, binaryMask);

    //fa una prima segmentazione, già efficace solo nella foto con tutta la pasta (non mangiata)
    secondImage = removeDishes(secondImage,15);

    //sfrutto funzione usata per rimovere cose attorno al pane semplificata
    Mat second;
    int numColors = 105;
    int delta = 40;
    secondImage = removeColors(image,numColors,numColors,delta);

    Mat graySecond;
    cvtColor(secondImage, graySecond, COLOR_BGR2GRAY);

    //maschera di quello trovato fino a qua
    int thresholdValue = 1;
    Mat secondMask;
    threshold(graySecond, secondMask, thresholdValue, 255, THRESH_BINARY);


    //maschera più precisa anche se forse non cambia niente, da rivedere
    vector<vector<Point>> contours;
    findContours(secondMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat finalMask = Mat::zeros(secondMask.size(), CV_8UC1);
    drawContours(finalMask, contours, -1, Scalar(255), FILLED);
    Mat secondWithMask;
    secondImage.copyTo(secondWithMask, finalMask);
    secondWithMask = removeDishes(secondWithMask,19);


    //agisce sull'immagine a colori
    for(int i=0;i<image.rows;i++)
    {
        for(int j =0;j<image.cols;j++)
        {
            if(secondWithMask.at<Vec3b>(i,j)==Vec3b(0,0,0))
                secondo.at<Vec3b>(i,j) = Vec3b (0,0,0);
        }
    }


    //toglie un po di schifi e sughetto NB mi buca la prima
    Mat out_gray = secondo.clone();
    cvtColor(out_gray, out_gray, COLOR_BGR2GRAY);
    Mat kernelone = getStructuringElement(MORPH_RECT, Size(9, 9));
    morphologyEx(out_gray, out_gray, MORPH_OPEN, kernelone);


    //cvtColor(out_gray, out_gray, COLOR_BGR2GRAY);
    vector<vector<Point>> contours1;
    findContours(out_gray, contours1, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Crea una maschera bianca con l'interno del contorno riempito di bianco
    Mat maskWithContours = Mat::zeros(out_gray.size(), CV_8UC1);
    drawContours(maskWithContours, contours1, -1, Scalar(255), FILLED);


    //agisce sulla foto a colori
    for(int i=0;i<image.rows;i++)
    {
        for(int j =0;j<image.cols;j++)
        {
            if(maskWithContours.at<unsigned char>(i,j)==0)
                secondo.at<Vec3b>(i,j) = Vec3b (0,0,0);
        }
    }

    //cerco di tappare i buchi
    Mat kerneltwo = getStructuringElement(MORPH_RECT, Size(133, 133));
    //dilate(maskWithContours, maskWithContours, kerneltwo);
    morphologyEx(maskWithContours, maskWithContours, MORPH_CLOSE, kerneltwo);

    return maskWithContours;

}

//Function that calls segmentSecond and sets the pixels outside the mask to zero.
//Used to perform the final form of segmentation for second dish
Mat getSecond(const Mat& image)
{
    Mat temp = image.clone();
    Mat second = segmentSecond(temp);
    for(int i=0;i<temp.rows;i++)
    {
        for(int j =0;j<temp.cols;j++)
        {
            if(second.at<unsigned char>(i,j)==0)
            {
                temp.at<Vec3b>(i,j)[0] = 0;
                temp.at<Vec3b>(i,j)[1] = 0;
                temp.at<Vec3b>(i,j)[2] = 0;
            }
        }
    }

    temp = removeDishes(temp,15); //15 è una buona via di mezzo rispetto al 25
    //return temp;
    temp = removeColors(temp,10,10,70);

    //temp = drawBox(temp,Scalar(255,0,0));
    return temp;
}

//Kmeans
Mat K_Means(Mat input, int k){
    Mat samples(input.rows * input.cols, input.channels(), CV_32F);
    for(int y=0; y<input.rows; y++){
        for(int x=0; x<input.cols; x++){
            for(int z=0; z<input.channels(); z++){
                if(input.channels()==3){
                    samples.at<float>(y + x*input.rows, z) = input.at<Vec3b>(y,x)[z]; //for color image
                }else{
                    samples.at<float>(y + x*input.rows, z) = input.at<uchar>(y,x); //for grayscale image
                }
            }
        }
    }

    Mat labels;
    int attempts=5;
    Mat centers;
    kmeans(samples, k, labels, TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 10, 1.0), attempts, KMEANS_PP_CENTERS, centers);

    Mat new_image(input.size(), input.type());
    for(int y=0; y<input.rows; y++){
        for(int x=0; x<input.cols; x++){
            int clusteres_idx=labels.at<int>(y+x*input.rows, 0);
            if(input.channels()==3){
                for(int z=0; z<input.channels(); z++){
                    new_image.at<Vec3b>(y,x)[z] = centers.at<float>(clusteres_idx, z);
                }
            }else{
                new_image.at<uchar>(y,x) = centers.at<float>(clusteres_idx, 0);
            }
        }
    }

    return new_image;
}

//Function that displays colored boxes around various dishes, salad, bread (if any), also allows access to images of individual
//food inside the box and box values (coordinates, height,width)
vector<Mat> viewBoxOnImage(vector<Mat>firstTray,vector<Mat>saladTray,vector<Mat>secondTray,vector<Mat>breadTray,vector<Mat>tray,vector<Mat>output,vector<Mat>&boxFirst,vector<Mat>&boxSecond,vector<Mat>&boxSalad,vector<Mat>&boxBread,std::vector<cv::Rect>&boxRectFirst,std::vector<cv::Rect>&boxRectSecond,std::vector<cv::Rect>&boxRectSalad,std::vector<cv::Rect>&boxRectBread,int n)
{

    Scalar red(0,0,255);
    Scalar blue(255,0,0);
    Scalar green(0,255,0);
    Scalar black(0,0,0);
    Scalar white(255,255,255);
    Scalar marine(255, 255, 0);


    for(int i=0;i<firstTray.size();i++)
    {
        Rect rectaF;
        Mat first = getFirst(firstTray[i]);
        first = drawBox(first,red,rectaF);
        boxFirst.push_back(first);
        boxRectFirst.push_back(rectaF);

        Rect rectaS;
        Mat salad = segmentSalad(saladTray[i]);
        salad = drawBox(salad,green,rectaS);
        boxSalad.push_back(salad);
        boxRectSalad.push_back(rectaS);
        for(int j=0;j<first.rows;j++)
        {
            for(int k = 0;k<first.cols;k++)
            {
                if(first.at<Vec3b>(j,k)[0]==red[0] && first.at<Vec3b>(j,k)[1]==red[1] &&first.at<Vec3b>(j,k)[2]==red[2])
                {
                    tray[i].at<Vec3b>(j,k)[0]=red[0];
                    tray[i].at<Vec3b>(j,k)[1]=red[1];
                    tray[i].at<Vec3b>(j,k)[2]=red[2];
                }
                if(salad.at<Vec3b>(j,k)[0]==green[0] && salad.at<Vec3b>(j,k)[1]==green[1] &&salad.at<Vec3b>(j,k)[2]==green[2])
                {
                    tray[i].at<Vec3b>(j,k)[0]=green[0];
                    tray[i].at<Vec3b>(j,k)[1]=green[1];
                    tray[i].at<Vec3b>(j,k)[2]=green[2];
                }

            }
        }
    }
    for(int i=0;i<secondTray.size();i++)
    {
        Rect rectaSC;
        Mat second = getSecond(secondTray[i]);
        second = drawBox(second,blue,rectaSC);
        boxRectSecond.push_back(rectaSC);
        boxSecond.push_back(second);
        for(int j=0;j<second.rows;j++)
        {
            for(int k = 0;k<second.cols;k++)
            {
                if(second.at<Vec3b>(j,k)[0]==blue[0] && second.at<Vec3b>(j,k)[1]==blue[1] &&second.at<Vec3b>(j,k)[2]==blue[2])
                {
                    tray[i].at<Vec3b>(j,k)[0]=blue[0];
                    tray[i].at<Vec3b>(j,k)[1]=blue[1];
                    tray[i].at<Vec3b>(j,k)[2]=blue[2];
                }
            }
        }

    }
    if(n==1 || n == 4 || n==5)
    {
        for(int i=0;i<breadTray.size();i++)
        {
            Rect rectB;
            Mat bread = breadTray[i];
            bread = drawBox(bread,marine,rectB);
            boxRectBread.push_back(rectB);
            boxBread.push_back(bread);
            for(int j=0;j<bread.rows;j++)
            {
                for(int k = 0;k<bread.cols;k++)
                {
                    if(bread.at<Vec3b>(j,k)[0] == marine[0] && bread.at<Vec3b>(j, k)[1] == marine[1] && bread.at<Vec3b>(j, k)[2] == marine[2])
                    {
                        tray[i].at<Vec3b>(j,k)[0]=marine[0];
                        tray[i].at<Vec3b>(j,k)[1]=marine[1];
                        tray[i].at<Vec3b>(j,k)[2]=marine[2];
                    }
                }
            }


        }
    }
    for(int i=0;i<4;i++)
        output.push_back(tray[i]);
    return output;
}

//Funcation to get all the masks of an image
Mat getAllMask(Mat img,int nTray)
{


    Scalar red(0,0,255);
    Scalar blue(255,0,0);
    Scalar green(0,255,0);
    Scalar black(0,0,0);
    Scalar white(255,255,255);
    Scalar marine(255, 255, 0);


    Mat out;
    Mat image = img.clone();
    Mat first,second,salad;
    Mat bread(image.rows,image.cols,image.type(),Scalar(0));
    first = getFirst(image);
    second = getSecond(image);
    salad = segmentSalad(image);
    if(nTray==1 || nTray==4 || nTray==5)
    {   if(nTray==4)
            bread = detectBreadByHisto(image,nTray);
        else
            bread = detectBreadByHisto(image,nTray);
    }


//    for(int i=0;i< image.rows;i++)
//    {
//        for(int j=0;j<image.cols;j++)
//        {
//            if(first.at<Vec3b>(i,j)[0]!=0 && first.at<Vec3b>(i,j)[1]!=0 && first.at<Vec3b>(i,j)[2]!=0)
//            {
//                first.at<Vec3b>(i,j)[0] = red[0];
//                first.at<Vec3b>(i,j)[1] = red[1];
//                first.at<Vec3b>(i,j)[2] = red[2];
//            }
//
//            if(second.at<Vec3b>(i,j)[0]!=0 && second.at<Vec3b>(i,j)[1]!=0 && second.at<Vec3b>(i,j)[2]!=0)
//            {
//                second.at<Vec3b>(i,j)[0] = blue[0];
//                second.at<Vec3b>(i,j)[1] = blue[1];
//                second.at<Vec3b>(i,j)[2] = blue[2];
//            }
//
//            if(salad.at<Vec3b>(i,j)[0]!=0 && salad.at<Vec3b>(i,j)[1]!=0 && salad.at<Vec3b>(i,j)[2]!=0)
//            {
//                salad.at<Vec3b>(i,j)[0] = green[0];
//                salad.at<Vec3b>(i,j)[1] = green[1];
//                salad.at<Vec3b>(i,j)[2] = green[2];
//            }
//
//            if(bread.at<Vec3b>(i,j)[0]!=0 && bread.at<Vec3b>(i,j)[1]!=0 && bread.at<Vec3b>(i,j)[2]!=0)
//            {
//                bread.at<Vec3b>(i,j)[0] = marine[0];
//                bread.at<Vec3b>(i,j)[1] = marine[1];
//                bread.at<Vec3b>(i,j)[2] = marine[2];
//            }
//        }
//    }

    out = first + second + salad + bread;

    cvtColor(out,out,COLOR_BGR2GRAY);
    for(int i=0;i<out.rows;i++)
    {
        for(int j=0;j<out.cols;j++)
        {
            if(out.at<unsigned char>(i,j)!=0 && out.at<unsigned char>(i,j)!=0 &&out.at<unsigned char >(i,j)!=0)
            {
                out.at<unsigned char >(i,j) = 255;
                out.at<unsigned char >(i,j) = 255;
                out.at<unsigned char >(i,j) = 255;

            }
        }
    }

    return out;
}



