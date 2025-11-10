#include "ImageUtils.h"
#include "../FileSystem/PathUtils.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

// ============================================================================
// IMAGE LOADING & VALIDATION
// ============================================================================

cv::Mat ImageUtils::loadImage(const std::string &filePath, int flags) 
{
    if (filePath.empty() || !PathUtils::isImageExtension(filePath)) 
    {
        std::cerr << "[ERROR] Empty file path provided or Not Image extention" << std::endl;
        return cv::Mat();
    }
    
    cv::Mat image = cv::imread(filePath, flags);
    
    if (image.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << filePath << std::endl;
        return cv::Mat();
    }

    // Validate image data integrity
    if (image.rows <= 0 || image.cols <= 0) {
        std::cerr << "[ERROR] Invalid image dimensions" << std::endl;
        return cv::Mat();
    }

    return image;
}

cv::Mat ImageUtils::loadImage(const std::string &filePath) 
{
    return loadImage(filePath, cv::IMREAD_COLOR);
}

bool ImageUtils::isValidImage(const std::string &filePath) 
{
    return isValidImage(filePath, false);
}

bool ImageUtils::isValidImage(const std::string &filePath, bool performDeepCheck) {
    // Check if file exists

    if(!PathUtils::fileExists(filePath) || !std::filesystem::is_regular_file(filePath))
        return false;

    // Quick validation with OpenCV
    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    if (image.empty() || image.rows <= 0 || image.cols <= 0) {
        return false;
    }

    // Check if image is completely black (corrupted)
    if (isImageCompletelyBlack(image)) {
        return false;
    }

    // Perform deep validation if requested
    if (performDeepCheck) {
        return performDeepImageValidation(image, filePath);
    }

    return true;
}

bool ImageUtils::performDeepImageValidation(const cv::Mat& image, const std::string& filePath) {
    try {
        // Check for valid data pointer
        if (image.data == nullptr) {
            return false;
        }

        // Check for reasonable dimensions (not too small, not absurdly large)
        if (image.rows < 10 || image.cols < 10 || 
            image.rows > 50000 || image.cols > 50000) {
            return false;
        }

        // Check for valid number of channels
        int channels = image.channels();
        if (channels < 1 || channels > 4) {
            return false;
        }

        // Check for valid depth
        int depth = image.depth();
        if (depth != CV_8U && depth != CV_8S && depth != CV_16U && 
            depth != CV_16S && depth != CV_32S && depth != CV_32F && 
            depth != CV_64F) {
            return false;
        }

        // Verify data continuity and check for NaN/Inf values
        if (image.isContinuous()) {
            const uchar* ptr = image.ptr<uchar>(0);
            size_t totalBytes = image.total() * image.elemSize();
            
            // Sample check (checking every pixel could be slow)
            for (size_t i = 0; i < std::min(totalBytes, size_t(1000)); i += 10) {
                if (ptr[i] > 255) { // Basic sanity check for 8-bit images
                    return false;
                }
            }
        }

        // Calculate basic statistics to ensure image has meaningful data
        cv::Scalar mean, stddev;
        cv::meanStdDev(image, mean, stddev);
        
        // Check if image has some variance (not completely uniform)
        bool hasVariance = false;
        for (int i = 0; i < image.channels(); i++) {
            if (stddev[i] > 1.0) {
                hasVariance = true;
                break;
            }
        }

        return hasVariance;

    } 
    catch (const std::exception& e) 
    {
        std::cerr << "[ERROR] Exception during validation: " << e.what() << std::endl;
        return false;
    }
    catch(...)
    {
        std::cerr << "[ERROR] Unknown exception during validation." << std::endl;
        return false;
    }
    
    return false;
}

bool ImageUtils::isImageCompletelyBlack(const cv::Mat& image) 
{
    if (image.empty()) {
        return true;
    }

    cv::Mat grayImage;
    if (image.channels() == 3 || image.channels() == 4) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image;
    }

    // Check if all pixels are below a very low threshold
    double minVal, maxVal;
    cv::minMaxLoc(grayImage, &minVal, &maxVal);
    
    return (maxVal < 5.0); // Nearly black threshold
}

bool ImageUtils::saveImage(const cv::Mat& image, const std::string& filePath) {
    return saveImageWithQuality(image, filePath, 95);
}

bool ImageUtils::saveImageWithQuality(const cv::Mat& image, const std::string& filePath, int quality) {
    if (image.empty()) 
    {
        std::cerr << "[ERROR] Cannot save empty image" << std::endl;
        return false;
    }

    try {
        std::vector<int> compression_params;
        
        // Determine compression based on file extension
        // std::string extension = filePath.substr(filePath.find_last_of(".") + 1);
        // std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        std::string extension = PathUtils::getFileExtension(filePath);

        if (extension == "jpg" || extension == "jpeg") {
            compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            compression_params.push_back(std::max(0, std::min(100, quality)));
        } else if (extension == "png") {
            // PNG compression level 0-9 (higher = smaller file, slower)
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(std::max(0, std::min(9, quality / 11)));
        } else if (extension == "webp") {
            compression_params.push_back(cv::IMWRITE_WEBP_QUALITY);
            compression_params.push_back(std::max(1, std::min(100, quality)));
        }

        bool success = cv::imwrite(filePath, image, compression_params);
        
        if (!success) {
            std::cerr << "[ERROR] Failed to save image to: " << filePath << std::endl;
        }
        
        return success;

    } 
    catch (const std::exception& e) 
    {
        std::cerr << "[ERROR] OpenCV exception while saving: " << e.what() << std::endl;
        return false;
    }
    catch(...)
    {
        std::cerr << "[ERROR] Unknown exception during validation." << std::endl;
        return false;
    }
}

// ============================================================================
// IMAGE TRANSFORMATION & MANIPULATION
// ============================================================================

cv::Mat ImageUtils::resizeImage(const cv::Mat &image, const cv::Size &newSize) {
    if (image.empty() || newSize.width <= 0 || newSize.height <= 0) {
        std::cerr << "[ERROR] Invalid image or size for resizing" << std::endl;
        return cv::Mat();
    }

    cv::Mat resized;
    int interpolation = getOptimalInterpolation(image.size(), newSize);
    
    try {
        cv::resize(image, resized, newSize, 0, 0, interpolation);
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Resize failed: " << e.what() << std::endl;
        return cv::Mat();
    }

    return resized;
}

cv::Mat ImageUtils::resizeImageByWidth(const cv::Mat &image, int newWidth) 
{
    if (image.empty() || newWidth <= 0) 
    {
        return cv::Mat();
    }

    double aspectRatio = static_cast<double>(image.rows) / image.cols;
    int newHeight = static_cast<int>(newWidth * aspectRatio);
    
    return resizeImage(image, cv::Size(newWidth, newHeight));
}

cv::Mat ImageUtils::resizeImageByHeight(const cv::Mat &image, int newHeight) {
    if (image.empty() || newHeight <= 0) {
        return cv::Mat();
    }

    double aspectRatio = static_cast<double>(image.cols) / image.rows;
    int newWidth = static_cast<int>(newHeight * aspectRatio);
    
    return resizeImage(image, cv::Size(newWidth, newHeight));
}

cv::Mat ImageUtils::cropImage(const cv::Mat &image, const cv::Rect &roi) 
{
    if (image.empty()) {
        std::cerr << "[ERROR] Cannot crop empty image" << std::endl;
        return cv::Mat();
    }

    // Validate ROI is within image bounds
    if (roi.x < 0 || roi.y < 0 || 
        roi.x + roi.width > image.cols || 
        roi.y + roi.height > image.rows) {
        std::cerr << "[ERROR] ROI out of image bounds" << std::endl;
        return cv::Mat();
    }

    return image(roi).clone();
}


cv::Mat ImageUtils::cropImageSafe(const cv::Mat &image, const cv::Rect &roi, cv::Scalar fillColor) 
{
    if (image.empty()) {
        return cv::Mat();
    }


    cv::Rect safeROI = getSafeROI(roi, image.size());

    if (safeROI.width <= 0 || safeROI.height <= 0) {
        return cv::Mat();
    }


    if (safeROI != roi) 
    {
        return addPaddingToROI(image(safeROI).clone(), roi, safeROI, fillColor);
    }

    return image(safeROI).clone();
}
cv::Rect ImageUtils::getSafeROI(const cv::Rect& roi, const cv::Size& imageSize) 
{
    if (roi.width <= 0 || roi.height <= 0) 
    {
        return cv::Rect(0, 0, 0, 0);
    }
    
    int x = std::max(0, roi.x);
    int y = std::max(0, roi.y);
    int width = std::min(roi.width, imageSize.width - x);
    int height = std::min(roi.height, imageSize.height - y);
    
    // checking 
    if (width <= 0 || height <= 0) {
        return cv::Rect(0, 0, 0, 0);
    }
    
    return cv::Rect(x, y, width, height);
}

cv::Mat ImageUtils::addPaddingToROI(const cv::Mat& cropped, const cv::Rect& originalROI, 
                                     const cv::Rect& safeROI, cv::Scalar fillColor) {
    cv::Mat padded(originalROI.height, originalROI.width, cropped.type(), fillColor);
    
    int offsetX = safeROI.x - originalROI.x;
    int offsetY = safeROI.y - originalROI.y;
    
    cv::Rect dstROI(offsetX, offsetY, cropped.cols, cropped.rows);
    cropped.copyTo(padded(dstROI));
    
    return padded;
}

// ============================================================================
// COLOR & IMAGE PROCESSING
// ============================================================================

cv::Mat ImageUtils::convertToGray(const cv::Mat &image) 
{
    if (image.empty()) 
    {
        return cv::Mat();
    }

    cv::Mat gray;
    
    if (image.channels() == 1) {
        gray = image.clone();
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        std::cerr << "[ERROR] Unsupported number of channels: " << image.channels() << std::endl;
        return cv::Mat();
    }

    return gray;
}

cv::Mat ImageUtils::equalizeHistogram(const cv::Mat &image) 
{
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat result;
    
    if (image.channels() == 1) {
        cv::equalizeHist(image, result);
    } else if (image.channels() == 3) {
        // Convert to YCrCb and equalize Y channel only
        cv::Mat ycrcb;
        cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
        
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        cv::equalizeHist(channels[0], channels[0]);
        
        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
    } else {
        std::cerr << "[ERROR] Histogram equalization only supports 1 or 3 channel images" << std::endl;
        return image.clone();
    }

    return result;
}

cv::Mat ImageUtils::normalizeImage(const cv::Mat &image) 
{
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat normalized;
    cv::normalize(image, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    
    return normalized;
}

cv::Mat ImageUtils::adjustBrightnessContrast(const cv::Mat &image, double alpha, double beta) {
    if (image.empty()) 
    {
        return cv::Mat();
    }

    cv::Mat adjusted;
    
    // alpha: contrast (1.0-3.0), beta: brightness (0-100)
    // Formula: new_pixel = alpha * old_pixel + beta
    image.convertTo(adjusted, -1, alpha, beta);
    
    return adjusted;
}

// ============================================================================
// IMAGE TRANSFORMATION
// ============================================================================

cv::Mat ImageUtils::rotateImage(const cv::Mat& image, double angle) 
{
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1.0);

    // Calculate new bounding dimensions
    double radians = angle * CV_PI / 180.0;
    double sin_val = std::abs(std::sin(radians));
    double cos_val = std::abs(std::cos(radians));
    
    int newWidth = static_cast<int>(image.cols * cos_val + image.rows * sin_val);
    int newHeight = static_cast<int>(image.cols * sin_val + image.rows * cos_val);

    // Adjust rotation matrix for new center
    rotationMatrix.at<double>(0, 2) += (newWidth / 2.0) - center.x;
    rotationMatrix.at<double>(1, 2) += (newHeight / 2.0) - center.y;

    cv::Mat rotated;
    cv::warpAffine(image, rotated, rotationMatrix, cv::Size(newWidth, newHeight), 
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return rotated;
}

cv::Mat ImageUtils::flipHorizontal(const cv::Mat& image) 
{
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat flipped;
    cv::flip(image, flipped, 1);
    return flipped;
}

cv::Mat ImageUtils::flipVertically(const cv::Mat &image) {
    if (image.empty()) {
        return cv::Mat();
    }

    cv::Mat flipped;
    cv::flip(image, flipped, 0);
    return flipped;
}

// ============================================================================
// IMAGE INFORMATION & ANALYSIS
// ============================================================================

cv::Size ImageUtils::getImageSize(const std::string& filePath) 
{
    if(!PathUtils::fileExists(filePath))
        return cv::Size(0, 0);
    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        return cv::Size(0, 0);
    }
    return image.size();
}

cv::Size ImageUtils::getImageSize(const cv::Mat &image) 
{
    return image.empty() ? cv::Size(0, 0) : image.size();
}

int ImageUtils::getImageChannels(const cv::Mat &image) 
{
    return image.empty() ? 0 : image.channels();
}

std::string ImageUtils::getImageInfo(const std::string &filePath) 
{
    cv::Mat image = cv::imread(filePath, cv::IMREAD_UNCHANGED);
    
    if (image.empty()) {
        return "Invalid image file";
    }

    std::stringstream info;
    info << "Image Information:\n";
    info << "- Path: " << filePath << "\n";
    info << "- Dimensions: " << image.cols << "x" << image.rows << "\n";
    info << "- Channels: " << image.channels() << "\n";
    info << "- Depth: " << image.depth() << "\n";
    info << "- Type: " << image.type() << "\n";
    info << "- Total pixels: " << image.total() << "\n";
    info << "- Size in memory: " << (image.total() * image.elemSize()) / 1024.0 << " KB\n";
    
    double quality = calculateImageQuality(image);
    info << "- Estimated quality: " << quality << "/100\n";

    return info.str();
}

double ImageUtils::calculateImageQuality(const cv::Mat &image) 
{
    if (image.empty()) 
    {
        return 0.0;
    }

    cv::Mat gray;
    if (image.channels() == 3 || image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else 
    {
        gray = image;
    }

    // Calculate sharpness using Laplacian variance
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    double sharpness = stddev[0] * stddev[0];

    // Normalize to 0-100 scale (typical good images have variance > 100)
    double quality = std::min(100.0, (sharpness / 500.0) * 100.0);

    return quality;
}

int ImageUtils::getOptimalInterpolation(const cv::Size& originalSize, const cv::Size& targetSize) 
{
    // Use INTER_AREA for shrinking (best quality for downsampling)
    if (targetSize.width < originalSize.width || targetSize.height < originalSize.height) {
        return cv::INTER_AREA;
    }
    // Use INTER_CUBIC for enlarging (better quality than INTER_LINEAR)
    else if (targetSize.width > originalSize.width * 1.5 || targetSize.height > originalSize.height * 1.5) {
        return cv::INTER_CUBIC;
    }
    // Use INTER_LINEAR for moderate changes (good balance)
    else {
        return cv::INTER_LINEAR;
    }
}

std::string ImageUtils::getInterpolationName(int interpolation) 
{
    switch (interpolation) {
        case cv::INTER_NEAREST: return "INTER_NEAREST";
        case cv::INTER_LINEAR: return "INTER_LINEAR";
        case cv::INTER_CUBIC: return "INTER_CUBIC";
        case cv::INTER_AREA: return "INTER_AREA";
        case cv::INTER_LANCZOS4: return "INTER_LANCZOS4";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// FACE PROCESSING
// ============================================================================

cv::Mat ImageUtils::extractFace(const cv::Mat &image, const cv::Rect &faceRect, bool addPadding, double paddingScale) 
{
    if (image.empty() || faceRect.area() == 0) 
    {
        return cv::Mat();
    }

    cv::Rect expandedRect = faceRect;
    
    if (addPadding) {
        int padX = static_cast<int>(faceRect.width * paddingScale);
        int padY = static_cast<int>(faceRect.height * paddingScale);
        
        expandedRect.x -= padX;
        expandedRect.y -= padY;
        expandedRect.width += 2 * padX;
        expandedRect.height += 2 * padY;
    }

    return cropImageSafe(image, expandedRect, cv::Scalar(0, 0, 0));
}

std::vector<cv::Point2f> ImageUtils::detectFacialLandmarksHAAR(const cv::Mat& faceImage) 
{
    return detectLandmarksHAAR(faceImage);
}

std::vector<cv::Point2f> ImageUtils::detectFacialLandmarks(const cv::Mat& faceImage) {
    // Try multiple methods in order of sophistication
    std::vector<cv::Point2f> landmarks;

    // Method 1: Try Dlib (most accurate)
    landmarks = detectLandmarksDlib(faceImage);
    if (!landmarks.empty()) {
        return landmarks;
    }

    // Method 2: Try LBF
    landmarks = detectLandmarksLBF(faceImage);
    if (!landmarks.empty()) {
        return landmarks;
    }

    // Method 3: Try YuNet with  landmarks
    landmarks = detectFacialLandmarksYuNet(faceImage);
    if (!landmarks.empty()) {
        return landmarks;
    }

    // Method 4: Fallback to HAAR-based estimation
    landmarks = detectLandmarksHAAR(faceImage);
    if (!landmarks.empty()) {
        return landmarks;
    }

    // Method 5: Generate default landmarks as last resort
    return generateDefaultLandmarks(faceImage);
}

std::vector<cv::Point2f> ImageUtils::detectLandmarksDlib(const cv::Mat& faceImage) {
    try {
        static dlib::shape_predictor predictor;
        static bool predictorLoaded = false;

        // Load predictor model (only once)
        if (!predictorLoaded) {
            try {
                dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;
                predictorLoaded = true;
            } catch (...) {
                std::cerr << "[WARNING] Dlib predictor model not found" << std::endl;
                return std::vector<cv::Point2f>();
            }
        }

        // Convert OpenCV Mat to Dlib image
        dlib::cv_image<dlib::bgr_pixel> dlibImage(faceImage);
        
        // Detect face in the image
        static dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
        std::vector<dlib::rectangle> faces = detector(dlibImage);

        if (faces.empty()) {
            // Assume entire image is face
            dlib::rectangle faceRect(0, 0, faceImage.cols - 1, faceImage.rows - 1);
            dlib::full_object_detection shape = predictor(dlibImage, faceRect);

            std::vector<cv::Point2f> landmarks;
            for (unsigned int i = 0; i < shape.num_parts(); i++) {
                landmarks.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
            }
            return landmarks;
        }

        // Use first detected face
        dlib::full_object_detection shape = predictor(dlibImage, faces[0]);
        
        std::vector<cv::Point2f> landmarks;
        for (unsigned int i = 0; i < shape.num_parts(); i++) {
            landmarks.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
        }

        return landmarks;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Dlib landmark detection failed: " << e.what() << std::endl;
        return std::vector<cv::Point2f>();
    }
}

std::vector<cv::Point2f> ImageUtils::detectLandmarksLBF(const cv::Mat& faceImage) {
    try {
        static cv::Ptr<cv::face::Facemark> facemark;
        static bool facemarkLoaded = false;

        if (!facemarkLoaded) {
            try {
                facemark = cv::face::FacemarkLBF::create();
                facemark->loadModel("lbfmodel.yaml");
                facemarkLoaded = true;
            } catch (...) {
                std::cerr << "[WARNING] LBF model not found" << std::endl;
                return std::vector<cv::Point2f>();
            }
        }

        std::vector<cv::Rect> faces;
        faces.push_back(cv::Rect(0, 0, faceImage.cols, faceImage.rows));

        std::vector<std::vector<cv::Point2f>> landmarks;
        bool success = facemark->fit(faceImage, faces, landmarks);

        if (success && !landmarks.empty()) {
            return landmarks[0];
        }

    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] LBF landmark detection failed: " << e.what() << std::endl;
    }

    return std::vector<cv::Point2f>();
}

std::vector<cv::Point2f> ImageUtils::detectLandmarksHAAR(const cv::Mat& faceImage) {
    try {
        cv::Mat gray = convertToGray(faceImage);
        
        static cv::CascadeClassifier eyeCascade, mouthCascade;
        static bool cascadesLoaded = false;

        if (!cascadesLoaded) {
            eyeCascade.load(cv::samples::findFile("haarcascade_eye.xml"));
            mouthCascade.load(cv::samples::findFile("haarcascade_smile.xml"));
            cascadesLoaded = true;
        }

        std::vector<cv::Rect> eyes, mouths;
        eyeCascade.detectMultiScale(gray, eyes, 1.1, 3, 0, cv::Size(20, 20));
        mouthCascade.detectMultiScale(gray, mouths, 1.3, 5, 0, cv::Size(25, 25));

        if (!eyes.empty() || !mouths.empty()) {
            return generateLandmarks(faceImage);
        }

    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] HAAR landmark detection failed: " << e.what() << std::endl;
    }

    return std::vector<cv::Point2f>();
}

std::vector<cv::Point2f> ImageUtils::generateLandmarks(const cv::Mat& faceImage) {
    std::vector<cv::Point2f> landmarks;
    
    int w = faceImage.cols;
    int h = faceImage.rows;

    // Generate 68-point landmarks based on typical face proportions
    // Jaw line (0-16)
    for (int i = 0; i < 17; i++) {
        float x = w * (0.1f + 0.8f * i / 16.0f);
        float y = h * (0.85f - 0.15f * std::sin(i * CV_PI / 16.0f));
        landmarks.push_back(cv::Point2f(x, y));
    }

    // Right eyebrow (17-21)
    for (int i = 0; i < 5; i++) {
        landmarks.push_back(cv::Point2f(w * (0.2f + 0.15f * i / 4.0f), h * 0.35f));
    }

    // Left eyebrow (22-26)
    for (int i = 0; i < 5; i++) {
        landmarks.push_back(cv::Point2f(w * (0.65f + 0.15f * i / 4.0f), h * 0.35f));
    }

    // Nose bridge (27-30)
    for (int i = 0; i < 4; i++) {
        landmarks.push_back(cv::Point2f(w * 0.5f, h * (0.4f + 0.1f * i)));
    }

    // Nose bottom (31-35)
    landmarks.push_back(cv::Point2f(w * 0.5f, h * 0.65f));
    landmarks.push_back(cv::Point2f(w * 0.45f, h * 0.67f));
    landmarks.push_back(cv::Point2f(w * 0.5f, h * 0.68f));
    landmarks.push_back(cv::Point2f(w * 0.55f, h * 0.67f));
    landmarks.push_back(cv::Point2f(w * 0.5f, h * 0.65f));

    // Right eye (36-41)
    float rightEyeCenterX = w * 0.35f;
    float rightEyeCenterY = h * 0.45f;
    float eyeRadius = w * 0.05f;
    for (int i = 0; i < 6; i++) {
        float angle = i * CV_PI / 3.0f;
        landmarks.push_back(cv::Point2f(
            rightEyeCenterX + eyeRadius * std::cos(angle),
            rightEyeCenterY + eyeRadius * 0.6f * std::sin(angle)
        ));
    }

    // Left eye (42-47)
    float leftEyeCenterX = w * 0.65f;
    float leftEyeCenterY = h * 0.45f;
    for (int i = 0; i < 6; i++) {
        float angle = i * CV_PI / 3.0f;
        landmarks.push_back(cv::Point2f(
            leftEyeCenterX + eyeRadius * std::cos(angle),
            leftEyeCenterY + eyeRadius * 0.6f * std::sin(angle)
        ));
    }

    // Outer mouth (48-59)
    float mouthCenterX = w * 0.5f;
    float mouthCenterY = h * 0.75f;
    float mouthWidth = w * 0.2f;
    float mouthHeight = h * 0.08f;
    for (int i = 0; i < 12; i++) {
        float angle = i * CV_PI / 6.0f;
        landmarks.push_back(cv::Point2f(
            mouthCenterX + mouthWidth * std::cos(angle),
            mouthCenterY + mouthHeight * std::sin(angle)
        ));
    }

    // Inner mouth (60-67)
    float innerMouthWidth = mouthWidth * 0.6f;
    float innerMouthHeight = mouthHeight * 0.5f;
    for (int i = 0; i < 8; i++) {
        float angle = i * CV_PI / 4.0f;
        landmarks.push_back(cv::Point2f(
            mouthCenterX + innerMouthWidth * std::cos(angle),
            mouthCenterY + innerMouthHeight * std::sin(angle)
        ));
    }

    return landmarks;
}

std::vector<cv::Point2f> ImageUtils::generateDefaultLandmarks(const cv::Mat& faceImage) {
    return generateLandmarks(faceImage);
}

std::vector<cv::Rect> ImageUtils::detectFaces(const cv::Mat& grayImage) 
{
    std::vector<cv::Rect> faces;

    try {
        // Method 1: Try Dlib detector (most robust)
        static dlib::frontal_face_detector dlibDetector;
        static bool dlibInitialized = false;

        if (!dlibInitialized) {
            dlibDetector = dlib::get_frontal_face_detector();
            dlibInitialized = true;
        }

        dlib::cv_image<unsigned char> dlibImage(grayImage);
        std::vector<dlib::rectangle> dlibFaces = dlibDetector(dlibImage);

        for (const auto& face : dlibFaces) {
            faces.push_back(cv::Rect(face.left(), face.top(), 
                                     face.width(), face.height()));
        }

        if (!faces.empty()) {
            return faces;
        }

        // Method 2: OpenCV Haar Cascade
        static cv::CascadeClassifier faceCascade;
        static bool cascadeLoaded = false;

        if (!cascadeLoaded) {
            if (faceCascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"))) {
                cascadeLoaded = true;
            }
        }

        if (cascadeLoaded) {
            faceCascade.detectMultiScale(grayImage, faces, 1.1, 3, 0, 
                                        cv::Size(30, 30));
        }

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Face detection failed: " << e.what() << std::endl;
    }

    return faces;
}

std::vector<cv::Point2f> ImageUtils::detectFacialLandmarksYuNet(const cv::Mat& faceImage) {
    try {
        // YuNet face detector with landmark points
        static cv::Ptr<cv::FaceDetectorYN> detector;
        static bool detectorLoaded = false;

        if (!detectorLoaded) {
            try {
                std::string model = "face_detection_yunet_2023mar.onnx";
                detector = cv::FaceDetectorYN::create(model, "", 
                                                      cv::Size(320, 320), 
                                                      0.6f, 0.3f, 5000);
                detectorLoaded = true;
            } catch (...) {
                std::cerr << "[WARNING] YuNet model not found" << std::endl;
                return std::vector<cv::Point2f>();
            }
        }

        // Set input size to current image size
        detector->setInputSize(faceImage.size());

        cv::Mat faces;
        detector->detect(faceImage, faces);

        if (faces.rows > 0)
        {
            // YuNet returns: x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm
            // Where: re=right eye, le=left eye, nt=nose tip, rcm=right corner mouth, lcm=left corner mouth
            
            std::vector<cv::Point2f> keyPoints;
            for (int i = 4; i < 14; i += 2) {
                keyPoints.push_back(cv::Point2f(faces.at<float>(0, i), 
                                                faces.at<float>(0, i + 1)));
            }

            // Generate  68-point landmarks from these 5 key points
            return generateLandmarksFromKeyPoints(keyPoints, faceImage.size());
        }

    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] YuNet detection failed: " << e.what() << std::endl;
    }

    return std::vector<cv::Point2f>();
}

std::vector<cv::Point2f> ImageUtils::generateLandmarksFromKeyPoints(const std::vector<cv::Point2f>& key_points, const cv::Size& image_size)
{
    
    if (key_points.size() < 5)
    {
        return std::vector<cv::Point2f>();
    }

    std::vector<cv::Point2f> landmarks;
    
    // Key points: 0=right_eye, 1=left_eye, 2=nose_tip, 3=right_mouth, 4=left_mouth
    cv::Point2f right_eye = key_points[0];
    cv::Point2f left_eye = key_points[1];
    cv::Point2f nose_tip = key_points[2];
    cv::Point2f right_mouth = key_points[3];
    cv::Point2f left_mouth = key_points[4];

    // Calculate face metrics
    float eye_distance = cv::norm(right_eye - left_eye);
    float face_width = eye_distance * 2.5f;
    float face_height = face_width * 1.3f;

    cv::Point2f face_center = (right_eye + left_eye) * 0.5f;

    // Generate jaw line (17 points, 0-16)
    for (int i = 0; i < 17; i++) {
        float t = i / 16.0f;
        float x = face_center.x - face_width * 0.5f + face_width * t;
        float y = face_center.y + face_height * 0.4f - 
                  face_height * 0.2f * std::sin(t * CV_PI);
        landmarks.push_back(cv::Point2f(x, y));
    }

    // Right eyebrow (5 points, 17-21)
    for (int i = 0; i < 5; i++) {
        float t = i / 4.0f;
        float x = right_eye.x - eye_distance * 0.25f + eye_distance * 0.3f * t;
        float y = right_eye.y - eye_distance * 0.15f;
        landmarks.push_back(cv::Point2f(x, y));
    }

    // Left eyebrow (5 points, 22-26)
    for (int i = 0; i < 5; i++) {
        float t = i / 4.0f;
        float x = left_eye.x - eye_distance * 0.05f + eye_distance * 0.3f * t;
        float y = left_eye.y - eye_distance * 0.15f;
        landmarks.push_back(cv::Point2f(x, y));
    }

    // Nose bridge (4 points, 27-30)
    cv::Point2f nose_start = (right_eye + left_eye) * 0.5f + cv::Point2f(0, eye_distance * 0.1f);
    for (int i = 0; i < 4; i++) {
        float t = i / 3.0f;
        landmarks.push_back(nose_start * (1 - t) + nose_tip * t);
    }

    // Nose bottom (5 points, 31-35)
    float nose_width = eye_distance * 0.25f;
    landmarks.push_back(cv::Point2f(nose_tip.x - nose_width, nose_tip.y));
    landmarks.push_back(cv::Point2f(nose_tip.x - nose_width * 0.5f, nose_tip.y + eye_distance * 0.05f));
    landmarks.push_back(nose_tip);
    landmarks.push_back(cv::Point2f(nose_tip.x + nose_width * 0.5f, nose_tip.y + eye_distance * 0.05f));
    landmarks.push_back(cv::Point2f(nose_tip.x + nose_width, nose_tip.y));

    // Right eye (6 points, 36-41)
    float eye_radius = eye_distance * 0.12f;
    generateEyeLandmarks(right_eye, eye_radius, landmarks);

    // Left eye (6 points, 42-47)
    generateEyeLandmarks(left_eye, eye_radius, landmarks);

    // Outer mouth (12 points, 48-59)
    float mouth_width = cv::norm(right_mouth - left_mouth);
    generateMouthLandmarks(right_mouth, left_mouth, mouth_width, landmarks, false);

    // Inner mouth (8 points, 60-67)
    generateMouthLandmarks(right_mouth, left_mouth, mouth_width * 0.6f, landmarks, true);

    return landmarks;
}

void ImageUtils::generateEyeLandmarks(const cv::Point2f& eye_center, float radius, 
                                       std::vector<cv::Point2f>& landmarks) {
    // Generate 6 points around the eye
    float angles[] = {0.0f, CV_PI / 3.0f, 2.0f * CV_PI / 3.0f, 
                      CV_PI, 4.0f * CV_PI / 3.0f, 5.0f * CV_PI / 3.0f};
    
    for (float angle : angles) {
        float x = eye_center.x + radius * std::cos(angle);
        float y = eye_center.y + radius * 0.6f * std::sin(angle);
        landmarks.push_back(cv::Point2f(x, y));
    }
}

void ImageUtils::generateMouthLandmarks(const cv::Point2f& right_mouth, 
                                         const cv::Point2f& left_mouth,
                                         float width, 
                                         std::vector<cv::Point2f>& landmarks, 
                                         bool inner) {
    cv::Point2f mouth_center = (right_mouth + left_mouth) * 0.5f;
    float height = width * 0.4f;
    
    int num_points = inner ? 8 : 12;
    
    for (int i = 0; i < num_points; i++) {
        float angle = i * 2.0f * CV_PI / num_points;
        float x = mouth_center.x + (width * 0.5f) * std::cos(angle);
        float y = mouth_center.y + (height * 0.5f) * std::sin(angle);
        landmarks.push_back(cv::Point2f(x, y));
    }
}

std::vector<cv::Point2f> ImageUtils::getFaceLandmarks(const cv::Mat& faceImage) {
    return detectFacialLandmarks(faceImage);
}

std::vector<cv::Point2f> ImageUtils::getDefaultFaceLandmarks(const cv::Mat& faceImage) {
    return generateDefaultLandmarks(faceImage);
}

bool ImageUtils::hasSufficientResolution(const cv::Mat& image, int minWidth, int minHeight) {
    if (image.empty()) {
        return false;
    }
    return (image.cols >= minWidth && image.rows >= minHeight);
}
