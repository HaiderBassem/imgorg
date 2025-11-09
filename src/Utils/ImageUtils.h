#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <dlib/image_processing.h>
#include <vector>
#include <string>

class ImageUtils {
public:
    // Image Loading & Validation
    static cv::Mat loadImage(const std::string &filePath, int flags);
    static cv::Mat loadImage(const std::string &filePath);
    static bool isValidImage(const std::string &filePath);
    static bool isValidImage(const std::string &filePath, bool performDeepCheck);
    static bool performDeepImageValidation(const cv::Mat& image, const std::string& filePath);
    static bool isImageCompletelyBlack(const cv::Mat& image);
    static bool saveImage(const cv::Mat& image, const std::string& filePath);
    static bool saveImageWithQuality(const cv::Mat& image, const std::string& filePath, int quality);

    // Image Transformation & Manipulation
    static cv::Mat resizeImage(const cv::Mat &image, const cv::Size &newSize);
    static cv::Mat resizeImageByWidth(const cv::Mat &image, int newWidth);
    static cv::Mat resizeImageByHeight(const cv::Mat &image, int newHeight);
    static cv::Mat cropImage(const cv::Mat &image, const cv::Rect &roi);
    static cv::Mat cropImageSafe(const cv::Mat &image, const cv::Rect &roi, cv::Scalar fillColor = cv::Scalar(0,0,0));
    static cv::Rect getSafeROI(const cv::Rect& roi, const cv::Size& imageSize);
    static cv::Mat addPaddingToROI(const cv::Mat& cropped, const cv::Rect& originalROI, const cv::Rect& safeROI, cv::Scalar fillColor);

    // Color & Image Processing
    static cv::Mat convertToGray(const cv::Mat &image);
    static cv::Mat equalizeHistogram(const cv::Mat &image);
    static cv::Mat normalizeImage(const cv::Mat &image);
    static cv::Mat adjustBrightnessContrast(const cv::Mat &image, double alpha, double beta);

    // Image Transformation
    static cv::Mat rotateImage(const cv::Mat& image, double angle);
    static cv::Mat flipHorizontal(const cv::Mat& image);
    static cv::Mat flipVertically(const cv::Mat &image);

    // Image Information & Analysis
    static cv::Size getImageSize(const std::string& filePath);
    static cv::Size getImageSize(const cv::Mat &image);
    static int getImageChannels(const cv::Mat &image);
    static std::string getImageInfo(const std::string &filePath);
    static double calculateImageQuality(const cv::Mat &image);
    static int getOptimalInterpolation(const cv::Size& originalSize, const cv::Size& targetSize);
    static std::string getInterpolationName(int interpolation);

    // Face Processing
    static cv::Mat extractFace(const cv::Mat &image, const cv::Rect &faceRect, bool addPadding = true, double paddingScale = 0.1);
    static std::vector<cv::Point2f> detectFacialLandmarksHAAR(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> detectFacialLandmarks(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> detectLandmarksDlib(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> detectLandmarksLBF(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> detectLandmarksHAAR(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> generateEnhancedLandmarks(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> generateDefaultLandmarks(const cv::Mat& faceImage);
    static std::vector<cv::Rect> detectFacesEnhanced(const cv::Mat& grayImage);
    static std::vector<cv::Point2f> detectFacialLandmarksYuNet(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> generateLandmarksFromKeyPoints(const std::vector<cv::Point2f>& key_points, const cv::Size& image_size);
    static void generateEyeLandmarks(const cv::Point2f& eye_center, float radius, std::vector<cv::Point2f>& landmarks);
    static void generateMouthLandmarks(const cv::Point2f& right_mouth, const cv::Point2f& left_mouth,float width, std::vector<cv::Point2f>& landmarks, bool inner);
    static std::vector<cv::Point2f> getEnhancedFaceLandmarks(const cv::Mat& faceImage);
    static std::vector<cv::Point2f> getDefaultFaceLandmarks(const cv::Mat& faceImage);
    static bool hasSufficientResolution(const cv::Mat& image, int minWidth, int minHeight);
};
