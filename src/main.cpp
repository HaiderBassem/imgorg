#include "Utils/ImageUtils.h"  

int main() 
{

    auto img = ImageUtils::loadImage("/home/cpluspluser/Pictures/Games/Alan Wake.jpg", cv::IMREAD_COLOR);
    std::cout<< ImageUtils::isValidImage("/home/cpluspluser/Desktop/Screenshot_20251103_204820.png",1);
    return 0;
}
