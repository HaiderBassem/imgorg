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
#ifdef DEBUG_MODE
        Logger::instance().debug("Retrying to load image as COLOR: " + filePath);
#endif
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
#ifdef DEBUG_MODE
    Logger::instance().debug("Loaded image: " + filePath + 
                           " [Size: " + std::to_string(image.cols) + "x" + 
                           std::to_string(image.rows) + ", Channels: " + 
                           std::to_string(image.channels()) + ", Mode: " + mode + "]");
#endif
    
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
        Logger::instance().warning("File does not exist: " + filePath);
        return false;
    }

    const std::string ext = PathUtils::getFileExtension(filePath);
    if (!PathUtils::isImageExtension(ext)) 
    {
        Logger::instance().warning("File extension not supported: " + filePath);
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
#ifdef DEBUG_MODE

    Logger::instance().debug("Image is valid: " + filePath + 
                           " [Size: " + std::to_string(testImage.cols) + "x" + 
                           std::to_string(testImage.rows) + "]");
#endif
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
        Logger::instance().warning("Invalid number of channels: " + filePath);
        return false;
    }


    int depth = image.depth();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F) 
    {
        Logger::instance().warning("Unsupported image depth: " + filePath);
        return false;
    }

    if (isImageCompletelyBlack(image)) 
    {
        Logger::instance().warning("Image appears to be completely black: " + filePath);
        return false;
    }

    if (image.cols < 10 || image.rows < 10) 
    {
        Logger::instance().warning("Image too small: " + filePath);
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

bool ImageUtils::saveImage(const cv::Mat& image, const std::string& filePath)
{
    if (image.empty()) {
        Logger::instance().error("Cannot save empty image to: " + filePath);
        return false;
    }


    std::string absolutePath = PathUtils::getAbsolutePath(filePath);
    std::string parentDir = PathUtils::getParentDirectory(absolutePath);


    if (!parentDir.empty() && 
        parentDir != "/" && 
        parentDir != "/home" && 
        !PathUtils::fileExists(parentDir)) {
        
        if (!PathUtils::createDirectoryIfNotExists(parentDir)) {
            Logger::instance().error("Failed to create directory: " + parentDir);
            return false;
        }
    }


    bool success = cv::imwrite(absolutePath, image);
    
    if (!success) {
        Logger::instance().error("Failed to save image to: " + absolutePath);
        return false;
    }


    if (!PathUtils::fileExists(absolutePath)) {
        Logger::instance().error("File verification failed: " + absolutePath);
        return false;
    }

    Logger::instance().info("Image saved: " + absolutePath);
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
#ifdef DEBUG_MODE
        Logger::instance().debug("Image already at target size, returning copy");
#endif
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
#ifdef DEBUG_MODE
Logger::instance().debug("Resized image: " + 
                           std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                           " -> " + std::to_string(newSize.width) + "x" + 
                           std::to_string(newSize.height) + 
                           " [Interpolation: " + getInterpolationName(interpolation) + "]");
#endif
    
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
     #ifdef DEBUG_MODE
      Logger::instance().debug("Cropped image: " + 
                           std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                           " -> " + std::to_string(roi.width) + "x" + 
                           std::to_string(roi.height) + 
                           " at (" + std::to_string(roi.x) + "," + 
                           std::to_string(roi.y) + ")");
    #endif
    return croppedImage;
}

cv::Mat ImageUtils::ropImageSafe(const cv::Mat &image, const cv::Rect &roi, cv::Scalar fillColor)
{
    if(image.empty())
    {
        Logger::instance().error("Connot crop empty image");
        return cv::Mat();
    }
    cv::Rect safeRoi = ImageUtils::getSafeROI(roi, image.size());

    if(safeRoi.width <= 0 || safeRoi.height <= 0)
    {
        Logger::instance().warning("ROI completely outside image bounds, creating filled image");
        
        cv::Mat filledImage(roi.height, roi.width, image.type(), fillColor);
        Logger::instance().debug("Created filled image: " + 
                std::to_string(roi.width) + "x" + 
                std::to_string(roi.height));
        return filledImage;
    }

    cv::Mat cropped;
    try
    {
        cropped = image(safeRoi).clone();
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("OpenCV crop failed: " + std::string(e.what()));
        return cv::Mat();
    }
    
    if(safeRoi != roi)
        cropped = ImageUtils::addPaddingToROI(cropped, roi, safeRoi, fillColor);
    
    if(cropped.empty())
    {
        Logger::instance().error("Crop operation produced empty image");
        return cv::Mat();
    }
     #ifdef DEBUG_MODE
    Logger::instance().debug("Safe crop completed: " + 
            std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
            " -> " + std::to_string(roi.width) + "x" + 
            std::to_string(roi.height) + 
            " [Safe ROI: " + std::to_string(safeRoi.x) + "," + 
            std::to_string(safeRoi.y) + " " + 
            std::to_string(safeRoi.width) + "x" + 
            std::to_string(safeRoi.height) + "]");
            #endif

    return cropped;
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

cv::Mat ImageUtils::convertToGray(const cv::Mat &image)
{
    if (image.empty()) 
    {
        Logger::instance().error("Cannot convert empty image to grayscale");
        return cv::Mat();
    }
    if(image.channels() == 1)
    {
        #ifdef DEBUG_MODE
        Logger::instance().debug("Image is already grayscale, returning copy");
        #endif
        return image.clone();

    }

    cv::Mat grayImage;
    try
    {
        if(image.channels() == 3)
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        else if (image.channels() == 4)
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        else 
        {
            Logger::instance().warning("Unsupported number of channels: " + std::to_string(image.channels()));
            return cv::Mat();
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("OpenCV grayscale conversion failed: " + 
                               std::string(e.what()));
        return cv::Mat();
    }
    
    if(grayImage.empty())
    {
        Logger::instance().error("Grayscale conversion produced empty image");
        return cv::Mat();
    }
        #ifdef DEBUG_MODE
    Logger::instance().debug("Converted to grayscale: " + 
                           std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                           " [" + std::to_string(image.channels()) + "ch -> 1ch]");
        #endif
    
    return grayImage;

}


cv::Mat ImageUtils::equalizeHistogram(const cv::Mat &image)
{
    if(image.empty())
    {
        Logger::instance().error("Cannot equalize histogram of empty image");
        return cv::Mat();
    }
    cv::Mat equlized;
    try
    {
        cv::Mat grayImage;
        if(image.channels() == 3) 
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        else
            grayImage = image.clone();
        
        cv::equalizeHist(grayImage, equlized);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Histogram equalization failed: " + std::string(e.what()));
        return cv::Mat();
    }
    
    if(equlized.empty())
    {
        Logger::instance().error("Histogram equalization produced empty image");
        return cv::Mat();
    }
         #ifdef DEBUG_MODE
        Logger::instance().debug("Equalized image histogram: " + 
                       std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                       " [Range: 0-255]");
        #endif

    return equlized;
}

cv::Mat ImageUtils::normalizeImage(const cv::Mat &image)
{
    if (image.empty()) 
    {
        Logger::instance().error("Cannot normalize empty image");
        return cv::Mat();
    }

    cv::Mat normalized;
    try
    {
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32F);

        cv::normalize(floatImage, normalized, 0.0, 1.0, cv::NORM_MINMAX);
        normalized.convertTo(normalized, CV_8U, 255.0);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Image normalization failed: " + std::string(e.what()));
        return cv::Mat();
    }

    if(normalized.empty())
    {
       Logger::instance().error("Normalization produced empty image");
        return cv::Mat();
    }
        #ifdef DEBUG_MODE
        Logger::instance().debug("Normalized image: " + 
                           std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                           " [Range: 0-255]");
        #endif
    
    return normalized;
}

cv::Mat ImageUtils::adjustBrightnessContrast(const cv::Mat &image, double alpha, double beta)
{
    if(image.empty())
    {
        Logger::instance().error("Cannot adjust brightness/contrast of empty image");
        return cv::Mat();
    }
    cv::Mat adjusted;
    try
    {
        image.convertTo(adjusted, -1, alpha, beta);
        adjusted = cv::max(0, cv::min(255, adjusted)); 
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Brightness/contrast adjustment failed: " + std::string(e.what()));
        return cv::Mat();
    }
    if(adjusted.empty())
    {
        Logger::instance().error("Brightness/contrast adjustment produced empty image");
        return cv::Mat();
    }
    #ifdef DEBUG_MODE
    Logger::instance().debug("Adjusted brightness/contrast: " + 
            std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
            " [Alpha: " + std::to_string(alpha) + 
            ", Beta: " + std::to_string(beta) + "]");
    #endif
    return adjusted;
    
}
cv::Mat ImageUtils::rotateImage(const cv::Mat& image, double angle)
{
    if (image.empty()) 
    {
        Logger::instance().error("Cannot rotate empty image");
        return cv::Mat();
    }

    cv::Mat rotated;
    try
    {

        cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);

        cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

        double radians = angle * CV_PI / 180.0;
        double sinVal = std::abs(std::sin(radians));
        double cosVal = std::abs(std::cos(radians));
        
        int newWidth = int(image.rows * sinVal + image.cols * cosVal);
        int newHeight = int(image.rows * cosVal + image.cols * sinVal);

        rotationMatrix.at<double>(0, 2) += (newWidth / 2.0) - center.x;
        rotationMatrix.at<double>(1, 2) += (newHeight / 2.0) - center.y;

        cv::warpAffine(image, rotated, rotationMatrix, cv::Size(newWidth, newHeight),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Image rotation failed: " + std::string(e.what()));
        return cv::Mat();
    }

    if(rotated.empty())
    {
        Logger::instance().error("Rotation produced empty image");
        return cv::Mat();
    }
    #ifdef DEBUG_MODE
    Logger::instance().debug("Rotated image: " + 
                       std::to_string(image.cols) + "x" + std::to_string(image.rows) + 
                       " -> " + std::to_string(rotated.cols) + "x" + std::to_string(rotated.rows) +
                       " [Angle: " + std::to_string(angle) + "°]");
    #endif
    return rotated;
}

// ex: if the image input has text hello, the output will be olleh
cv::Mat ImageUtils::flipHorizontally(const cv::Mat& image)
{
    if (image.empty()) 
    {
        Logger::instance().error("Cannot flip empty image");
        return cv::Mat();
    }

    cv::Mat flipped;
    try
    {

        cv::flip(image, flipped, 1);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Horizontal flip failed: " + std::string(e.what()));
        return cv::Mat();
    }

    if(flipped.empty())
    {
        Logger::instance().error("Horizontal flip produced empty image");
        return cv::Mat();
    }
     #ifdef DEBUG_MODE
    Logger::instance().debug("Horizontally flipped image: " + 
                       std::to_string(image.cols) + "x" + std::to_string(image.rows));
    #endif
    
    return flipped;
}

cv::Mat ImageUtils::flipVertically(const cv::Mat &image)
{
    if(image.empty())
    {
        Logger::instance().error("Cannot flip empty image");
        return cv::Mat();
    }

    cv::Mat flipped;
    try
    {
        cv::flip(image, flipped, 0);
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Vertical flip failed: " + std::string(e.what()));
        return cv::Mat();
    }
    if(flipped.empty())
    {
        Logger::instance().error("Vertical flip produced empty image");
        return cv::Mat();
    }
    Logger::instance().debug("Vertivally fliped image: " + std::to_string(image.cols) + "x" + std::to_string(image.rows));

    
    return flipped;
}

cv::Size ImageUtils::getImageSize(const std::string &filePath)
{
     if (filePath.empty()) 
    {
        Logger::instance().error("Cannot get size of empty file path");
        return cv::Size();
    }
    try
    {
        cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED | cv::IMREAD_IGNORE_ORIENTATION);
        if(image.empty())
        {
            Logger::instance().error("Failed to load image: "+ filePath );
            return cv::Size();
        }
        cv::Size size = image.size();
        #ifdef DEBUG_MODE
        Logger::instance().debug("Image Size: " +filePath + " - " + std::to_string(size.width) + "x" + std::to_string(size.height));
        #endif
        return size; 
    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Failed to get image size: " + filePath + " - " + std::string(e.what()));
        return cv::Size();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Error reading file: " + filePath + " - " + std::string(e.what()));
        return cv::Size();
    }
    

}

cv::Size ImageUtils::getImageSize(const cv::Mat &image)
{
    if(image.empty())
    {
        Logger::instance().error("Cannot get size of empty image");
        return cv::Size();
    }

    try
    {
        cv::Size size = image.size();
        #ifdef DEBUG_MODE
        Logger::instance().debug("Image size: " + 
                    std::to_string(size.width) + "x" + std::to_string(size.height) +
                    " [Channels: " + std::to_string(image.channels()) + "]");
        #endif
        return size;

    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("error occured while program is running: " + std::string(e.what()));
    }
        catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("error occured while program is running: " + std::string(e.what()));
    }
    
}

int ImageUtils::getImageChannels(const cv::Mat &image)
{
    if(image.empty())
    {
        Logger::instance().warning("Cannot get number of empty Image channels");
        return -1;
    }

    try
    {
        
        int ChannelsCount = image.channels();
#ifdef DEBUG_MODE
        Logger::instance().debug("Image channels: " + std::to_string(ChannelsCount) +
                           " [Size: " + std::to_string(image.cols) + "x" + 
                           std::to_string(image.rows) + "]");
#endif
        return ChannelsCount;


    }
    catch(const cv::Exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Error getting image channels: " + std::string(e.what()));
        return -1;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        Logger::instance().error("Error getting image channels: " + std::string(e.what()));
        return -1;
    }
    
    return 0;
}

// std::string ImageUtils::getImageInfo(const std::string &filePath)
// {
//     return std::string();
// }

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


    cv::Mat padded;
    cv::copyMakeBorder(cropped, padded, 
                      padTop, padBottom, padLeft, padRight, 
                      cv::BORDER_CONSTANT, fillColor);
#ifdef DEBUG_MODE
    Logger::instance().debug("Added padding to ROI: " + std::string(
                           "L:" + std::to_string(padLeft) + " R:" + std::to_string(padRight) +
                           " T:" + std::to_string(padTop) + " B:" + std::to_string(padBottom)));
#endif    
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