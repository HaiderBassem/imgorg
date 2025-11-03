#include "ImageUtils.h"
#include "../Utils/Logger.h"
#include "../FileSystem/PathUtils.h"
#include <iostream>

cv::Mat ImageUtils::loadImage(const std::string &filePath, int flags)
{

    if(!PathUtils::fileExists(filePath))
    {
        Logger::instance().error("Image file does not exist: " + filePath);
        return cv::Mat();
    }


    const std::string ext = PathUtils::getFileExtension(filePath);
    if(!PathUtils::isImageExtension(ext))
    {
        Logger::instance().warning("File may not be a supported image format: " + filePath);
    }


    cv::Mat image = cv::imread(filePath, flags);


    if(image.empty() && flags != cv::IMREAD_COLOR)
    {
        Logger::instance().debug("Retrying to load image as COLOR: " + filePath);
        image = cv::imread(filePath, cv::IMREAD_COLOR);
    }

 
    if(image.empty())
    {
        Logger::instance().error("Failed to load image: " + filePath);
        return cv::Mat();
    }


    std::string mode;
    switch(flags) {
        case cv::IMREAD_GRAYSCALE: mode = "GRAYSCALE"; break;
        case cv::IMREAD_UNCHANGED: mode = "UNCHANGED"; break;
        default: mode = "COLOR";
    }
    
    Logger::instance().debug("Loaded image: " + filePath + 
                           " [Size: " + std::to_string(image.cols) + "x" + 
                           std::to_string(image.rows) + ", Channels: " + 
                           std::to_string(image.channels()) + ", Mode: " + mode + "]");
    
    return image;
}

cv::Mat ImageUtils::loadImage(const std::string &filePath)
{
    return loadImage(filePath, cv::IMREAD_COLOR);
}


bool ImageUtils::isValidImage(const std::string& filePath)
{

    if (!PathUtils::fileExists(filePath)) 
    {
        Logger::instance().debug("File does not exist: " + filePath);
        return false;
    }

    const std::string ext = PathUtils::getFileExtension(filePath);
    if (!PathUtils::isImageExtension(ext)) 
    {
        Logger::instance().debug("File extension not supported: " + filePath);
        return false;
    }


    cv::Mat testImage = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    if (testImage.empty()) 
    {
        Logger::instance().debug("OpenCV failed to read image: " + filePath);
        return false;
    }


    if (testImage.cols <= 0 || testImage.rows <= 0) 
    {
        Logger::instance().debug("Image has invalid dimensions: " + filePath);
        return false;
    }

    Logger::instance().debug("Image is valid: " + filePath + 
                           " [Size: " + std::to_string(testImage.cols) + "x" + 
                           std::to_string(testImage.rows) + "]");
    return true;
}

bool ImageUtils::isValidImage(const std::string& filePath, bool performDeepCheck)
{
    if (!PathUtils::fileExists(filePath)) 
        return false;
    

    const std::string ext = PathUtils::getFileExtension(filePath);
    if (!PathUtils::isImageExtension(ext)) 
        return false;


    cv::Mat testImage = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    if (testImage.empty()) 
        return false;
    

    if (testImage.cols <= 0 || testImage.rows <= 0) 
        return false;
    

    if (performDeepCheck) 
        return performDeepImageValidation(testImage, filePath);


    return true;
}

bool ImageUtils::performDeepImageValidation(const cv::Mat& image, const std::string& filePath)
{

    if (image.channels() != 1 && image.channels() != 3 && image.channels() != 4) 
    {
        Logger::instance().debug("Invalid number of channels: " + filePath);
        return false;
    }


    int depth = image.depth();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) 
    {
        Logger::instance().debug("Unsupported image depth: " + filePath);
        return false;
    }

    if (isImageCompletelyBlack(image)) 
    {
        Logger::instance().debug("Image appears to be completely black: " + filePath);
        return false;
    }

    if (image.cols < 10 || image.rows < 10) 
    {
        Logger::instance().debug("Image too small: " + filePath);
        return false;
    }


    return true;
}

bool ImageUtils::isImageCompletelyBlack(const cv::Mat& image)
{
    if (image.empty()) return true;

    cv::Mat gray;
    if (image.channels() == 3) 
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) 
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else 
    {
        gray = image;
    }

    double minVal, maxVal;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    

    return maxVal == 0;
}

bool ImageUtils::saveImage(const cv::Mat image, const std::string &filePath)
{
    if(image.empty())
    {
        Logger::instance().warning("Connot save empty image to: " + filePath);
        return false;
    }
    std::string parentDir = PathUtils::getParentDirectory(filePath);
    if(!parentDir.empty() && !PathUtils::createDirectoryIfNotExists(parentDir))
    {
        Logger::instance().error("Failed to create directory: " + parentDir);
        return false;
    }

    bool success = cv::imwrite(filePath, image);
    if(!success)
    {
        Logger::instance().error("Failed to save image to: " + filePath);
        return false;
    }

    if (!PathUtils::fileExists(filePath)) 
    {
        Logger::instance().error("File was not created after save: " + filePath);
        return false;
    }
        Logger::instance().debug("Successfully saved image: " + filePath + 
                           " [Size: " + std::to_string(image.cols) + "x" + 
                           std::to_string(image.rows) + ", Channels: " + 
                           std::to_string(image.channels()) + "]");
    return true;
}

cv::Mat ImageUtils::resizeImage(const cv::Mat &image, const cv::Size &newSize)
{
    if(image.empty())
    {
        Logger::instance().warning("Cannot resize empty image");
        return cv::Mat();
    }

    if(image.size() == newSize)
    {
        Logger::instance().debug("Image already at target size, returning copy");
        return image.clone(); // return copy
    }

    int interpolation = ImageUtils::getOptimalInterpolation(image.size(), newSize);

    cv::Mat resizedImage;
    try
    {
        cv::resize(image, resizedImage, newSize, 0, 0, interpolation);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("OpenCV resize failed: " + std::string(e.what()));
        return cv::Mat();
    }

    if(resizedImage.empty())
    {
        Logger::instance().error("Resize operation produced empty image");
        return cv::Mat();
    }
    
Logger::instance().debug("Resized image: " + 
                           std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                           " -> " + std::to_string(newSize.width) + "x" + 
                           std::to_string(newSize.height) + 
                           " [Interpolation: " + getInterpolationName(interpolation) + "]");
    
    return resizedImage;
}

cv::Mat ImageUtils::resizeImageByWidth(const cv::Mat &image, int newWidth)
{
    if (image.empty() || newWidth <= 0) 
    {
        Logger::instance().error("Invalid parameters for resize by width");
        return cv::Mat();
    }

    double aspectRatio = static_cast<double>(image.rows) / image.cols;
    int newHight = static_cast<int> (newWidth * aspectRatio);

    return resizeImage(image, cv::Size(newWidth, newHight));
}



cv::Mat ImageUtils::resizeImageByHight(const cv::Mat &image, int newHeight)
{
    if (image.empty() || newHeight <= 0) 
    {
        Logger::instance().error("Invalid parameters for resize by height");
        return cv::Mat();
    }

    double aspectRatio = static_cast<double> (image.cols) / image.rows;
    int newWidth = static_cast<int>(newHeight * aspectRatio);
    return resizeImage(image, cv::Size(newWidth, newHeight));
}


int ImageUtils::getOptimalInterpolation(const cv::Size& originalSize, const cv::Size& targetSize)
{
    double scaleX = static_cast<double>(targetSize.width) / originalSize.width;
    double scaleY = static_cast<double>(targetSize.height) / originalSize.height;
    double scale = std::max(scaleX, scaleY);


    if (scale > 1.0) 
    {
 
        return cv::INTER_CUBIC;
    } else if (scale < 0.5) 
    {

        return cv::INTER_AREA;
    } else 
    {

        return cv::INTER_LINEAR;
    }
}

std::string ImageUtils::getInterpolationName(int interpolation)
{
    switch (interpolation) 
    {
        case cv::INTER_NEAREST: return "NEAREST";
        case cv::INTER_LINEAR: return "LINEAR";
        case cv::INTER_CUBIC: return "CUBIC";
        case cv::INTER_AREA: return "AREA";
        case cv::INTER_LANCZOS4: return "LANCZOS4";
        default: return "UNKNOWN";
    }
}