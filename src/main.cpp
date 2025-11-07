#include "Utils/ImageUtils.h"

int main()
{

    auto img = ImageUtils::loadImage("/home/cpluspluser/Desktop/photo_2025-11-01_21-21-09.jpg", cv::IMREAD_COLOR);
    ImageUtils::saveImage(ImageUtils::convertToGray(img), "/home/cpluspluser/Desktop/test.png");
    if(ImageUtils::isValidImage("/home/cpluspluser/Desktop/photo_2025-11-01_21-21-09.jpg",1))
        std::cout<<" \n\\n\ntest\n\n";
    else
        std::cout<< "\n\\n\nsad\n\n";

    return 0;

}
