#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>


namespace ImageUtils
{
    cv::Mat loadImage(const std::string &filePath);
    cv::Mat loadImage(const std::string &filePath, int flags);
    bool isValidImage(const std::string& filePath);
    bool isValidImage(const std::string& filePath, bool performDeepCheck);
    

    bool performDeepImageValidation(const cv::Mat& image, const std::string& filePath);
    bool isImageCompletelyBlack(const cv::Mat& image);
    bool saveImage(const cv::Mat& image, const std::string& filePath);


    cv::Mat resizeImage(const cv::Mat& image, const cv::Size& newSize);
    cv::Mat resizeImageByWidth(const cv::Mat& image, int newWidth);
    cv::Mat resizeImageByHight(const cv::Mat& iamge, int newHight);
    cv::Mat cropImage(const cv::Mat& image, const cv::Rect& roi);
    cv::Mat ropImageSafe(const cv::Mat& image, const cv::Rect& roi, cv::Scalar fillColor = cv::Scalar(0, 0, 0));
    cv::Mat addPaddingToROI(const cv::Mat& cropped, const cv::Rect& originalROI, const cv::Rect& safeROI, cv::Scalar fillColor);
    cv::Rect getSafeROI(const cv::Rect& roi, const cv::Size& imageSize);
    cv::Mat cropImageSafe(const cv::Mat& image, const cv::Rect& roi, cv::Scalar fillColor = cv::Scalar(0, 0, 0));

    // color manipulation
    cv::Mat convertToGray(const cv::Mat& image);
    cv::Mat equalizeHistogram(const cv::Mat& image);
    cv::Mat normalizeImage(const cv::Mat& iamge);
    cv::Mat adjustBrightnessContrast(const cv::Mat& image, double alpha, double beta);


    //rtoeation 
    cv::Mat rotateImage(const cv::Mat& iamge, double angle);
    cv::Mat flipHorizantal(const cv::Mat& iamge);
    cv::Mat flipVertical(const cv::Mat& image);


    // inforamtion of image 
    cv::Size getImageSize(const std::string& filePath);
    cv::Size getImageSize(const cv::Mat& image);
    int getImageChannels(const cv::Mat& image);
    std::string getImageInfo(const std::string& filePath);
    double calculateImageQuality(const cv::Mat& image);
    int getOptimalInterpolation(const cv::Size& originalSize, const cv::Size& targetSize);
    std::string getInterpolationName(int interpolation);


    // face Detector
    cv::Mat extractFace(const cv::Mat& image, const cv::Rect&  faceRect);
    cv::Mat alignFace(const cv::Mat& faceImage);
    cv::Mat preprocessFace(const cv::Mat& faceImage);

    // check quality
    bool isImageBlurry(const cv::Mat& image, double threshold = 100.0);
    bool hasSufficientResolution(const cv::Mat& image, int minWidth = 100, int minHight = 100);
    bool isPortraitOrientation(const cv::Mat& image);

    //tools
    cv::Mat createSquareCrop(const cv::Mat& iamge, const cv::Rect& roi);
    std::vector<cv::Mat> splitChannels(const cv::Mat& image);
    cv::Mat mergeChannels(const std::vector<cv::Mat>& channels);

    //saving 
    bool saveImageWithQuality(const cv::Mat& iamge, const std::string& filePath, int quality = 95);
    bool saveImageOptimized(const cv::Mat& image, const std::string& filePath);

    // check contant
    bool containsFaces(const cv::Mat& image); // I will use it later..
    bool isImageDark(const cv::Mat& image, double threshold = 0.3);

    // process more than one 
    std::vector<cv::Mat> loadImages(const std::vector<std::string>& filePaths);
    bool saveImages(const std::vector<cv::Mat>& images, const std::string& outputDir);

    cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize = 5);
    cv::Mat applyMedianBlur(const cv::Mat& iamge, int kernelSize = 5);
    cv::Mat applyBilateralFilter(const cv::Mat& iamge, int d = 9, double sigmaColor = 75, double sigmaSpace = 75);

    cv::Mat smartCrop(const cv::Mat& iamge, const cv::Size& targetSize);
    cv::Mat padToSquare(const cv::Mat& image, int targetSize = -1);

    double calculateSSIM(const cv::Mat& image1, cv::Mat& image2);
    double calculateMSE(const cv::Mat& image1, cv::Mat& image2);
    double calculatePSNR(const cv::Mat& image1, cv::Mat& image2);

    
} // namespace ImageUtils
