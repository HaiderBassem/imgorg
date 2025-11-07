#include "Utils/ImageUtils.h"  

int main() 
{

    auto img = ImageUtils::loadImage("/home/cpluspluser/Desktop/photo_2025-11-01_21-21-09.jpg", cv::IMREAD_COLOR);
    ImageUtils::saveImage(ImageUtils::convertToGray(img), "/home/cpluspluser/Desktop/test.png");
    //std::cout<< ImageUtils::isValidImage("/home/cpluspluser/Desktop/Screenshot_20251103_204820.png",1);
    return 0;
    
}
