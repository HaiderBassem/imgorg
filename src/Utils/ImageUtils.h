#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <dlib/image_processing.h>
#include <vector>
#include <string>

/**
 * @namespace ImageUtils
 * @brief A comprehensive utility namespace for image processing, face detection, and facial landmark detection.
 * 
 * This namespace provides a wide range of static methods for image loading, transformation, processing,
 * face detection, and facial landmark extraction using multiple algorithms including Dlib, OpenCV LBF,
 * and HAAR cascades.
 * 
 * @note All methods are static and thread-safe when used with separate image instances.
 */
namespace ImageUtils
{
    
    // ============================================================================
    // Image Loading & Validation
    // ============================================================================

    /**
     * @brief Loads an image from file with specified reading mode.
     * @param filePath Path to the image file.
     * @param flags OpenCV imread flags (e.g., cv::IMREAD_COLOR, cv::IMREAD_GRAYSCALE). Default is cv::IMREAD_COLOR.
     * @return cv::Mat Loaded image matrix. Returns empty matrix if loading fails.
     * @throws std::runtime_error if file doesn't exist or cannot be loaded.
     * 
     * @example
     * @code
     * cv::Mat colorImage = ImageUtils::loadImage("path/to/image.jpg", cv::IMREAD_COLOR);
     * cv::Mat grayImage = ImageUtils::loadImage("path/to/image.jpg", cv::IMREAD_GRAYSCALE);
     * @endcode
     */
    static cv::Mat loadImage(const std::string& filePath, int flags = cv::IMREAD_COLOR);

    /**
     * @brief Loads an image in default color mode (convenience overload).
     * @param filePath Path to the image file.
     * @return cv::Mat Loaded color image matrix.
     * 
     * @see loadImage(const std::string&, int)
     */
    static cv::Mat loadImage(const std::string& filePath);

    /**
     * @brief Validates if the file is a supported image format and can be loaded.
     * @param filePath Path to the image file.
     * @return true if file exists, has supported extension, and can be loaded as image.
     * @return false if any validation check fails.
     */
    static bool isValidImage(const std::string& filePath);

    /**
     * @brief Performs comprehensive image validation with optional deep checking.
     * @param filePath Path to the image file.
     * @param performDeepCheck If true, performs additional validation including channel count, depth, and content checks.
     * @return true if image passes all validation checks.
     * 
     * Deep check includes:
     * - Valid channel count (1, 3, or 4)
     * - Supported depth (CV_8U, CV_16U, CV_32F)
     * - Non-black image content
     * - Minimum size requirements
     */
    static bool isValidImage(const std::string &filePath, bool performDeepCheck);

    /**
     * @brief Performs deep validation on an already loaded image matrix.
     * @param image Image matrix to validate.
     * @param filePath Optional file path for logging purposes.
     * @return true if image meets all deep validation criteria.
     */
    static bool performDeepImageValidation(const cv::Mat& image, const std::string& filePath);

    /**
     * @brief Checks if an image is completely black (all pixel values are zero).
     * @param image Image matrix to check.
     * @return true if all pixel values are zero, false otherwise.
     */
    static bool isImageCompletelyBlack(const cv::Mat& image);

    /**
     * @brief Saves an image to the specified file path.
     * @param image Image matrix to save.
     * @param filePath Destination file path.
     * @return true if image was successfully saved, false otherwise.
     * @throws std::runtime_error if image is empty or directory creation fails.
     */
    static bool saveImage(const cv::Mat& image, const std::string& filePath);

    /**
     * @brief Saves an image with specified JPEG quality compression.
     * @param image Image matrix to save.
     * @param filePath Destination file path.
     * @param quality JPEG quality setting (0-100, higher means better quality).
     * @return true if image was successfully saved with quality setting.
     */
    static bool saveImageWithQuality(const cv::Mat& image, const std::string& filePath, int quality);

    // ============================================================================
    // Image Transformation & Manipulation
    // ============================================================================

    /**
     * @brief Resizes an image to the specified dimensions with optimal interpolation.
     * @param image Source image matrix.
     * @param newSize Target size (width, height).
     * @return cv::Mat Resized image matrix.
     * @throws std::invalid_argument if image is empty or newSize is invalid.
     */
    static cv::Mat resizeImage(const cv::Mat &image, const cv::Size &newSize);

    /**
     * @brief Resizes an image by specifying the new width while maintaining aspect ratio.
     * @param image Source image matrix.
     * @param newWidth Target width in pixels.
     * @return cv::Mat Resized image matrix with proportional height.
     */
    static cv::Mat resizeImageByWidth(const cv::Mat &image, int newWidth);

    /**
     * @brief Resizes an image by specifying the new height while maintaining aspect ratio.
     * @param image Source image matrix.
     * @param newHeight Target height in pixels.
     * @return cv::Mat Resized image matrix with proportional width.
     */
    static cv::Mat resizeImageByHeight(const cv::Mat &image, int newHeight);

    /**
     * @brief Crops a region of interest from an image.
     * @param image Source image matrix.
     * @param roi Region of interest rectangle (x, y, width, height).
     * @return cv::Mat Cropped image matrix.
     * @throws std::out_of_range if ROI is outside image boundaries.
     */
    static cv::Mat cropImage(const cv::Mat &image, const cv::Rect &roi);

    /**
     * @brief Safely crops a region with automatic boundary checking and padding.
     * @param image Source image matrix.
     * @param roi Desired region of interest.
     * @param fillColor Color used for padding when ROI extends outside image boundaries.
     * @return cv::Mat Cropped image with padding if needed.
     */
    static cv::Mat cropImageSafe(const cv::Mat &image, const cv::Rect &roi, cv::Scalar fillColor = cv::Scalar(0,0,0));

    /**
     * @brief Calculates a safe ROI that fits within image boundaries.
     * @param roi Desired region of interest.
     * @param imageSize Size of the source image.
     * @return cv::Rect Adjusted ROI that is guaranteed to be within image boundaries.
     */
    static cv::Rect getSafeROI(const cv::Rect& roi, const cv::Size& imageSize);

    /**
     * @brief Adds padding to a cropped image to match the original ROI dimensions.
     * @param cropped The cropped image matrix.
     * @param originalROI The originally requested ROI.
     * @param safeROI The actually cropped ROI (safe region).
     * @param fillColor Color used for padding.
     * @return cv::Mat Padded image matching original ROI dimensions.
     */
    static cv::Mat addPaddingToROI(const cv::Mat& cropped, const cv::Rect& originalROI, const cv::Rect& safeROI, cv::Scalar fillColor);

    // ============================================================================
    // Color & Image Processing
    // ============================================================================

    /**
     * @brief Converts an image to grayscale.
     * @param image Source image matrix (color or grayscale).
     * @return cv::Mat Grayscale image matrix.
     * @throws std::invalid_argument if image has unsupported channel count.
     */
    static cv::Mat convertToGray(const cv::Mat &image);

    /**
     * @brief Applies histogram equalization to enhance image contrast.
     * @param image Source image matrix.
     * @return cv::Mat Contrast-enhanced image matrix.
     */
    static cv::Mat equalizeHistogram(const cv::Mat &image);

    /**
     * @brief Normalizes image pixel values to [0, 255] range.
     * @param image Source image matrix.
     * @return cv::Mat Normalized image matrix.
     */
    static cv::Mat normalizeImage(const cv::Mat &image);

    /**
     * @brief Adjusts image brightness and contrast.
     * @param image Source image matrix.
     * @param alpha Contrast control (1.0 = no change, <1.0 = lower contrast, >1.0 = higher contrast).
     * @param beta Brightness control (0 = no change, positive = brighter, negative = darker).
     * @return cv::Mat Adjusted image matrix.
     */
    static cv::Mat adjustBrightnessContrast(const cv::Mat &image, double alpha, double beta);

    // ============================================================================
    // Image Transformation
    // ============================================================================

    /**
     * @brief Rotates an image by specified angle around center.
     * @param image Source image matrix.
     * @param angle Rotation angle in degrees (positive = counter-clockwise).
     * @return cv::Mat Rotated image matrix with adjusted canvas size.
     */
    static cv::Mat rotateImage(const cv::Mat& image, double angle);

    /**
     * @brief Flips an image horizontally (left-right).
     * @param image Source image matrix.
     * @return cv::Mat Horizontally flipped image matrix.
     */
    static cv::Mat flipHorizontal(const cv::Mat& image);

    /**
     * @brief Flips an image vertically (up-down).
     * @param image Source image matrix.
     * @return cv::Mat Vertically flipped image matrix.
     */
    static cv::Mat flipVertically(const cv::Mat &image);

    // ============================================================================
    // Image Information & Analysis
    // ============================================================================

    /**
     * @brief Gets the size of an image from file without fully loading it.
     * @param filePath Path to the image file.
     * @return cv::Size Image dimensions (width, height).
     */
    static cv::Size getImageSize(const std::string& filePath);

    /**
     * @brief Gets the size of a loaded image matrix.
     * @param image Image matrix.
     * @return cv::Size Image dimensions (width, height).
     */
    static cv::Size getImageSize(const cv::Mat &image);

    /**
     * @brief Gets the number of channels in an image.
     * @param image Image matrix.
     * @return int Number of channels (1=grayscale, 3=BGR, 4=BGRA).
     */
    static int getImageChannels(const cv::Mat &image);

    /**
     * @brief Generates comprehensive information about an image file.
     * @param filePath Path to the image file.
     * @return std::string Formatted string containing image properties.
     */
    static std::string getImageInfo(const std::string &filePath);

    /**
     * @brief Calculates a quality score for an image based on multiple factors.
     * @param image Image matrix to analyze.
     * @return double Quality score between 0.0 (poor) and 1.0 (excellent).
     * 
     * Quality factors include:
     * - Contrast (30%)
     * - Sharpness (40%)
     * - Noise level (20%)
     * - Brightness balance (10%)
     */
    static double calculateImageQuality(const cv::Mat &image);

    /**
     * @brief Determines optimal interpolation method based on scaling factor.
     * @param originalSize Source image dimensions.
     * @param targetSize Target image dimensions.
     * @return int OpenCV interpolation constant.
     * 
     * Algorithm:
     * - INTER_CUBIC for upscaling (>1.0x)
     * - INTER_AREA for significant downscaling (<0.5x)
     * - INTER_LINEAR for moderate scaling (0.5x - 1.0x)
     */
    static int getOptimalInterpolation(const cv::Size& originalSize, const cv::Size& targetSize);

    /**
     * @brief Converts interpolation constant to human-readable name.
     * @param interpolation OpenCV interpolation constant.
     * @return std::string Descriptive name of interpolation method.
     */
    static std::string getInterpolationName(int interpolation);

    // ============================================================================
    // Face Processing & Landmark Detection
    // ============================================================================

    /**
     * @brief Extracts face region from an image with optional padding.
     * @param image Source image containing face.
     * @param faceRect Bounding box of the face region.
     * @param addPadding If true, adds padding around the face region.
     * @param paddingScale Padding amount as fraction of face dimensions (default: 0.1 = 10%).
     * @return cv::Mat Extracted face image.
     */
    static cv::Mat extractFace(const cv::Mat &image, const cv::Rect &faceRect, bool addPadding = true, double paddingScale = 0.1);

    /**
     * @brief Main facial landmark detection function with multi-algorithm fallback.
     * @param faceImage Face image for landmark detection.
     * @return std::vector<cv::Point2f> Detected facial landmarks (68 points in Dlib format).
     * 
     * Detection sequence:
     * 1. Dlib (highest accuracy, 68 points)
     * 2. OpenCV LBF (high accuracy, 68 points)
     * 3. HAAR cascade (robust fallback, variable points)
     * 4. Enhanced generated landmarks (final fallback)
     */
    static std::vector<cv::Point2f> detectFacialLandmarks(const cv::Mat& faceImage);

    /**
     * @brief [Deprecated] Legacy HAAR-based landmark detection.
     * @param faceImage Face image for landmark detection.
     * @return std::vector<cv::Point2f> Detected facial landmarks.
     * @deprecated Use detectFacialLandmarks() instead for multi-algorithm approach.
     */
    static std::vector<cv::Point2f> detectFacialLandmarksHAAR(const cv::Mat& faceImage);

    /**
     * @brief Detects facial landmarks using Dlib's 68-point model.
     * @param faceImage Face image for landmark detection.
     * @return std::vector<cv::Point2f> 68 facial landmarks in standard Dlib order.
     * 
     * Requires: shape_predictor_68_face_landmarks.dat model file
     */
    static std::vector<cv::Point2f> detectLandmarksDlib(const cv::Mat& faceImage);

    /**
     * @brief Detects facial landmarks using OpenCV LBF algorithm.
     * @param faceImage Face image for landmark detection.
     * @return std::vector<cv::Point2f> 68 facial landmarks.
     * 
     * Requires: lbfmodel.yaml trained model
     */
    static std::vector<cv::Point2f> detectLandmarksLBF(const cv::Mat& faceImage);

    /**
     * @brief Detects facial landmarks using HAAR cascade approach.
     * @param faceImage Face image for landmark detection.
     * @return std::vector<cv::Point2f> Variable number of landmarks based on feature detection.
     */
    static std::vector<cv::Point2f> detectLandmarksHAAR(const cv::Mat& faceImage);

    /**
     * @brief Generates facial landmarks using geometric modeling.
     * @param faceImage Face image for landmark generation.
     * @return std::vector<cv::Point2f> 68 synthetically generated landmarks.
     * @details Generates 68-point landmarks based on typical face proportions:
     *  - Jaw line: points 0-16
     *  - Right eyebrow: points 17-21
     *  - Left eyebrow: points 22-26
     *  - Nose bridge: points 27-30
     *  - Nose bottom: points 31-35
     *  - Right eye: points 36-41
     *  - Left eye: points 42-47
     *  - Outer lip: points 48-59
     *  - Inner lip: points 60-67
     */
    static std::vector<cv::Point2f> generateLandmarks(const cv::Mat& faceImage);

    /**
     * @brief Generates basic default landmarks for fallback scenarios.
     * @param faceImage Face image for landmark generation.
     * @return std::vector<cv::Point2f> 4 basic facial landmarks (eyes, nose, mouth).
     */
    static std::vector<cv::Point2f> generateDefaultLandmarks(const cv::Mat& faceImage);

    /**
     * @brief face detection using multiple cascade classifiers.
     * @param grayImage Grayscale image for face detection.
     * @return std::vector<cv::Rect> Detected face bounding boxes, sorted by area (largest first).
     */
    static std::vector<cv::Rect> detectFaces(const cv::Mat& grayImage);

    /**
     * @brief Detects facial landmarks using YuNet deep learning model.
     * @param faceImage Face image for landmark detection.
     * @return std::vector<cv::Point2f> Facial landmarks generated from YuNet keypoints.
     * 
     * Requires: face_detection_yunet_2023mar.onnx model file
     */
    static std::vector<cv::Point2f> detectFacialLandmarksYuNet(const cv::Mat& faceImage);

    /**
     * @brief Generates full landmark set from key facial points.
     * @param key_points 5 key points from YuNet: [right_eye, left_eye, nose_tip, right_mouth, left_mouth].
     * @param image_size Size of the source image.
     * @return std::vector<cv::Point2f> 68 landmarks generated through geometric interpolation.
     */
    static std::vector<cv::Point2f> generateLandmarksFromKeyPoints(const std::vector<cv::Point2f>& key_points, const cv::Size& image_size);

    /**
     * @brief Generates circular eye landmarks around a center point.
     * @param eye_center Center point of the eye.
     * @param radius Radius of the eye region.
     * @param landmarks Output vector to append generated eye landmarks.
     */
    static void generateEyeLandmarks(const cv::Point2f& eye_center, float radius, std::vector<cv::Point2f>& landmarks);

    /**
     * @brief Generates mouth landmarks in elliptical pattern.
     * @param right_mouth Right corner of the mouth.
     * @param left_mouth Left corner of the mouth.
     * @param width Width of the mouth region.
     * @param landmarks Output vector to append generated mouth landmarks.
     * @param inner If true, generates inner mouth landmarks; otherwise outer mouth landmarks.
     */
    static void generateMouthLandmarks(const cv::Point2f& right_mouth, const cv::Point2f& left_mouth, float width, std::vector<cv::Point2f>& landmarks, bool inner);

    /**
     * @brief [Deprecated] Legacy landmarks generation.
     * @param faceImage Face image for landmark generation.
     * @return std::vector<cv::Point2f> facial landmarks.
     * @deprecated Use generateLandmarks() instead.
     */
    static std::vector<cv::Point2f> getFaceLandmarks(const cv::Mat& faceImage);

    /**
     * @brief [Deprecated] Legacy default landmarks generation.
     * @param faceImage Face image for landmark generation.
     * @return std::vector<cv::Point2f> Default facial landmarks.
     * @deprecated Use generateDefaultLandmarks() instead.
     */
    static std::vector<cv::Point2f> getDefaultFaceLandmarks(const cv::Mat& faceImage);

    /**
     * @brief Checks if image has sufficient resolution for processing.
     * @param image Image to check.
     * @param minWidth Minimum required width in pixels.
     * @param minHeight Minimum required height in pixels.
     * @return true if image meets or exceeds minimum resolution requirements.
     */
    static bool hasSufficientResolution(const cv::Mat& image, int minWidth, int minHeight);
} // ImageUtils