# Food-recognition-and-leftover-estimation
The developed system is able to
- recognize and localize all the food items in the tray images, considering the food categories detailed in the dataset
- segment each food item in the tray image to compute the corresponding food quantity (i.e., amount of pixels)
- compare the “before meal” and “after meal” images to find which food among the initial ones was eaten and which was not. The 
leftovers quantity is then estimated as the difference in the number of pixels of the food item in the pair of 
images.

For the time being, the system is not complete, lacking the segmentation of main courses from side dishes and the merging of the segmentation phase with the recognition phase (already developed). The system is running exclusively on the dataset provided in the various trays
