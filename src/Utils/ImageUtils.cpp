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

cv::Mat ImageUtils::cropImage(const cv::Mat &image, const cv::Rect &roi)
{
    if(image.empty())
    {
        Logger::instance().warning("Connot crop empty image");
        return cv::Mat();
    }

    if(roi.x < 0 || roi.y < 0 ||
       roi.x + roi.width > image.cols ||
       roi.y + roi.height > image.rows)
    {
        Logger::instance().error("ROI out of image bounds: " + std::string("Image(" + std::to_string(image.cols) + "x" + 
                        std::to_string(image.rows) + "), " +
                        "ROI(" + std::to_string(roi.x) + "," + 
                        std::to_string(roi.y) + " " + 
                        std::to_string(roi.width) + "x" + 
                        std::to_string(roi.height) + ")"));
        return cv::Mat();
    }

    if(roi.width <= 0 || roi.height <= 0)
    {
        Logger::instance().error("Invalid ROI dimensions: " + 
                        std::to_string(roi.width) + "x" + 
                        std::to_string(roi.height));
        return cv::Mat();

    }

    // cut the image 
    cv::Mat croppedImage;
    try
    {
        croppedImage = image(roi).clone(); // copy
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("OpenCV crop failed: " + std::string(e.what()));
        return cv::Mat();
    }
    
    if(croppedImage.empty())
    {
        Logger::instance().error("Crop operation produced empty image");
        return cv::Mat();
    }

      Logger::instance().debug("Cropped image: " + 
                           std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                           " -> " + std::to_string(roi.width) + "x" + 
                           std::to_string(roi.height) + 
                           " at (" + std::to_string(roi.x) + "," + 
                           std::to_string(roi.y) + ")");
    
    return croppedImage;
}

cv::Mat ImageUtils::cropImageSafe(const cv::Mat& image, const cv::Rect& roi, cv::Scalar fillColor)
{
    if (image.empty()) 
    {
        Logger::instance().error("Cannot crop empty image");
        return cv::Mat();
    }


    cv::Rect safeRoi = ImageUtils::getSafeROI(roi, image.size());
    

    if (safeRoi.width <= 0 || safeRoi.height <= 0) 
    {
        Logger::instance().error("ROI completely outside image bounds");
        return cv::Mat();
    }


    cv::Mat cropped = image(safeRoi).clone();

    if (safeRoi != roi) 
        cropped = ImageUtils::addPaddingToROI(cropped, roi, safeRoi, fillColor);

    return cropped;
}

cv::Rect ImageUtils::getSafeROI(const cv::Rect& roi, const cv::Size& imageSize)
{
    int x = std::max(0, roi.x);
    int y = std::max(0, roi.y);
    int width = std::min(roi.width, imageSize.width - x);
    int height = std::min(roi.height, imageSize.height - y);
    

    width = std::max(0, width);
    height = std::max(0, height);
    
    return cv::Rect(x, y, width, height);
}

cv::Mat ImageUtils::addPaddingToROI(const cv::Mat& cropped, const cv::Rect& originalROI, 
                                   const cv::Rect& safeROI, cv::Scalar fillColor)
{

    int padLeft = safeROI.x - originalROI.x;
    int padTop = safeROI.y - originalROI.y;
    int padRight = originalROI.width - safeROI.width - padLeft;
    int padBottom = originalROI.height - safeROI.height - padTop;

    // تطبيق التعبئة
    cv::Mat padded;
    cv::copyMakeBorder(cropped, padded, 
                      padTop, padBottom, padLeft, padRight, 
                      cv::BORDER_CONSTANT, fillColor);

    Logger::instance().debug("Added padding to ROI: " + std::string(
                           "L:" + std::to_string(padLeft) + " R:" + std::to_string(padRight) +
                           " T:" + std::to_string(padTop) + " B:" + std::to_string(padBottom)));
    
    return padded;
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