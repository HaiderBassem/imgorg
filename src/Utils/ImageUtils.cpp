#include "ImageUtils.h"
#include "../Utils/Logger.h"
#include "../FileSystem/PathUtils.h"
#include <iostream>

cv::Mat ImageUtils::loadImage(const std::string &filePath)
{
    if(!PathUtils::fileExists(filePath))
    {
        std::cerr <<"\033[31mImage file does not exists: \033[0m" <<filePath;
        Logger::instance().error("Image file does not exists" + filePath);
        return cv::Mat();
    }

    const std::string ext = PathUtils::getFileExetension(filePath);
    if(!PathUtils::isImageExetension(ext))
    {
        std::cerr<<"\033[33mFile is not a supported image format: \033[0m" << filePath;
    }
    cv::Mat image = cv::imread(filePath, cv::IMREAD_COLOR);

    if(image.empty())
    {
        std::cerr<<"\033[33mFailed to load iamge: \033[0m" << filePath;
        Logger::instance().warning("Failed to load iamge: " + filePath);
        return cv::Mat();
    }

    std::cout<< "\033[32mSuccessfully loaded image: " << filePath
            <<"[Size: " << std::to_string(image.cols) << "x" 
            <<std::to_string(image.rows) << ", Channels: " 
            <<std::to_string(image.channels()) << "]\033[0m";

    return image;
}