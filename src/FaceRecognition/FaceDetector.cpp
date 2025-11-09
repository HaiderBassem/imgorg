#include "FaceDetector.h"
#include "../Utils/ImageUtils.h"
#include<filesystem>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <thread>
#include <mutex>

// ============================================================================
// CONSTRUCTOR & DESTRUCTOR
// ============================================================================

FaceDetector::FaceDetector() 
    : initialized_(false), totalDetections_(0), totalConfidence_(0.0) {
    config_ = FaceDetectorConfig();
}

FaceDetector::FaceDetector(const FaceDetectorConfig& config)
    : config_(config), initialized_(false), totalDetections_(0), totalConfidence_(0.0) {
}

FaceDetector::~FaceDetector() {
    releaseModels();
}

// ============================================================================
// INITIALIZATION & MODEL MANAGEMENT
// ============================================================================

bool FaceDetector::initialize() {
    return initialize(config_);
}

bool FaceDetector::initialize(const FaceDetectorConfig& config) {
    std::lock_guard<std::mutex> lock(detectionMutex_);
    
    config_ = config;
    lastError_.clear();
    
    try {
        std::cout << "[INFO] Initializing FaceDetector..." << std::endl;
        
        bool success = loadModels();
        
        if (!success) {
            lastError_ = "Failed to load required models";
            initialized_ = false;
            return false;
        }
        
        initialized_ = true;
        std::cout << "[INFO] FaceDetector initialized successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        lastError_ = std::string("Initialization error: ") + e.what();
        initialized_ = false;
        return false;
    }
}

bool FaceDetector::isInitialized() const {
    return initialized_;
}

void FaceDetector::setConfig(const FaceDetectorConfig& config) {
    std::lock_guard<std::mutex> lock(detectionMutex_);
    config_ = config;
}

FaceDetectorConfig FaceDetector::getConfig() const {
    return config_;
}

bool FaceDetector::loadModels() {
    bool allSuccess = true;
    
    // Load models based on detection method
    switch (config_.detectionMethod) {
        case DetectionMethod::DLIB:
            allSuccess &= loadDlibModel();
            break;
        case DetectionMethod::HAAR_CASCADE:
        case DetectionMethod::LBP_CASCADE:
            break;
        case DetectionMethod::DNN_CAFFE:
        case DetectionMethod::DNN_TENSORFLOW:
            allSuccess &= loadDNNModel();
            break;
        case DetectionMethod::YUNET:
            allSuccess &= loadYuNetModel();
            break;
        case DetectionMethod::AUTO:
            loadDlibModel();
            loadYuNetModel();
            loadDNNModel();
            break;
    }
    
    // Load landmark detection models
    if (config_.detectLandmarks) {
        switch (config_.landmarkMethod) {
            case LandmarkMethod::DLIB_68:
                allSuccess &= loadDlibModel();
                break;
            case LandmarkMethod::LBF:
                allSuccess &= loadLBFModel();
                break;
            case LandmarkMethod::YUNET:
                allSuccess &= loadYuNetModel();
                break;
            default:
                break;
        }
    }
    
    return allSuccess;
}

bool FaceDetector::loadDlibModel() {
    try {
        if (!dlibDetector_) {
            dlibDetector_ = std::make_unique<dlib::frontal_face_detector>(
                dlib::get_frontal_face_detector()
            );
        }
        
        if (!dlibPredictor_ && config_.detectLandmarks) {
            dlibPredictor_ = std::make_unique<dlib::shape_predictor>();
            dlib::deserialize(config_.dlibModelPath) >> *dlibPredictor_;
            std::cout << "[INFO] Loaded Dlib model: " << config_.dlibModelPath << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[WARNING] Failed to load Dlib model: " << e.what() << std::endl;
        return false;
    }
}

bool FaceDetector::loadLBFModel() {
    try {
        lbfModel_ = cv::face::FacemarkLBF::create();
        lbfModel_->loadModel(config_.lbfModelPath);
        std::cout << "[INFO] Loaded LBF model: " << config_.lbfModelPath << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[WARNING] Failed to load LBF model: " << e.what() << std::endl;
        return false;
    }
}

bool FaceDetector::loadYuNetModel() {
    try {
        yunetDetector_ = cv::FaceDetectorYN::create(
            config_.yunetModelPath,
            "",
            cv::Size(320, 320),
            config_.minConfidence,
            0.3f,
            5000
        );
        std::cout << "[INFO] Loaded YuNet model: " << config_.yunetModelPath << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[WARNING] Failed to load YuNet model: " << e.what() << std::endl;
        return false;
    }
}

bool FaceDetector::loadDNNModel() {
    try {
        if (config_.detectionMethod == DetectionMethod::DNN_CAFFE) {
            dnnNet_ = cv::dnn::readNetFromCaffe(config_.caffeProtoPath, config_.caffeModelPath);
        } else if (config_.detectionMethod == DetectionMethod::DNN_TENSORFLOW) {
            dnnNet_ = cv::dnn::readNetFromTensorflow(config_.tfModelPath, config_.tfConfigPath);
        }
        
        if (config_.useGPU) {
            dnnNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            dnnNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            dnnNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            dnnNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        std::cout << "[INFO] Loaded DNN model successfully" << std::endl;
        return true;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[WARNING] Failed to load DNN model: " << e.what() << std::endl;
        return false;
    }
}

void FaceDetector::releaseModels() {
    std::lock_guard<std::mutex> lock(detectionMutex_);
    
    dlibDetector_.reset();
    dlibPredictor_.reset();
    lbfModel_.release();
    yunetDetector_.release();
    
    initialized_ = false;
}

// ============================================================================
// MAIN FACE DETECTION METHODS
// ============================================================================

std::vector<FaceDetectionResult> FaceDetector::detectFaces(const cv::Mat& image) {
    if (!initialized_) {
        std::cerr << "[ERROR] FaceDetector not initialized" << std::endl;
        return std::vector<FaceDetectionResult>();
    }
    
    if (image.empty()) {
        std::cerr << "[ERROR] Empty input image" << std::endl;
        return std::vector<FaceDetectionResult>();
    }
    
    std::lock_guard<std::mutex> lock(detectionMutex_);
    
    try {
        cv::Mat processedImage;
        preprocessImage(image, processedImage);
        
        std::vector<cv::Rect> faceRects;
        std::vector<float> confidences;
        
        switch (config_.detectionMethod) {
            case DetectionMethod::DLIB:
                faceRects = detectFacesDlib(processedImage);
                confidences.resize(faceRects.size(), 0.95f);
                break;
            case DetectionMethod::HAAR_CASCADE:
                faceRects = detectFacesHaar(processedImage);
                confidences.resize(faceRects.size(), 0.85f);
                break;
            case DetectionMethod::LBP_CASCADE:
                faceRects = detectFacesLBP(processedImage);
                confidences.resize(faceRects.size(), 0.80f);
                break;
            case DetectionMethod::DNN_CAFFE:
            case DetectionMethod::DNN_TENSORFLOW:
                faceRects = detectFacesDNN(processedImage);
                confidences.resize(faceRects.size(), 0.90f);
                break;
            case DetectionMethod::YUNET:
                faceRects = detectFacesYuNet(processedImage);
                confidences.resize(faceRects.size(), 0.92f);
                break;
            case DetectionMethod::AUTO:
                faceRects = detectFacesYuNet(processedImage);
                if (faceRects.empty()) {
                    faceRects = detectFacesDNN(processedImage);
                }
                if (faceRects.empty()) {
                    faceRects = detectFacesDlib(processedImage);
                }
                confidences.resize(faceRects.size(), 0.88f);
                break;
        }
        
        if (config_.filterOverlappingFaces && faceRects.size() > 1) {
            faceRects = filterOverlappingFaces(faceRects, confidences);
        }
        
        if (faceRects.size() > static_cast<size_t>(config_.maxFacesPerImage)) {
            faceRects.resize(config_.maxFacesPerImage);
        }
        
        std::vector<FaceDetectionResult> results;
        results.reserve(faceRects.size());
        
        for (size_t i = 0; i < faceRects.size(); ++i) {
            FaceDetectionResult result;
            result.boundingBox = faceRects[i];
            result.confidence = i < confidences.size() ? confidences[i] : 0.8f;
            result.detectionMethod = static_cast<int>(config_.detectionMethod);
            
            if (config_.extractFaces) {
                result.faceImage = extractFace(image, faceRects[i], true);
            }
            
            if (config_.detectLandmarks && !result.faceImage.empty()) {
                result.landmarks = detectLandmarks(result.faceImage, config_.landmarkMethod);
                
                if (!result.landmarks.empty()) {
                    cv::Point2f offset(result.boundingBox.x, result.boundingBox.y);
                    for (auto& pt : result.landmarks) {
                        pt += offset;
                    }
                }
            }
            
            if (config_.alignFaces && !result.landmarks.empty()) {
                result.alignedFace = alignFace(image, result.landmarks, config_.alignmentMethod);
            }
            
            results.push_back(result);
        }
        
        if (config_.sortByConfidence) {
            sortResultsByConfidence(results);
        }
        
        totalDetections_ += results.size();
        for (const auto& result : results) {
            totalConfidence_ += result.confidence;
        }
        
        lastResults_ = results;
        lastProcessedImage_ = image.clone();
        
        return results;
        
    } catch (const std::exception& e) {
        lastError_ = std::string("Detection error: ") + e.what();
        std::cerr << "[ERROR] " << lastError_ << std::endl;
        return std::vector<FaceDetectionResult>();
    }
}

std::vector<FaceDetectionResult> FaceDetector::detectFaces(const std::string& imagePath) {
    cv::Mat image = ImageUtils::loadImage(imagePath);
    
    if (image.empty()) {
        lastError_ = "Failed to load image: " + imagePath;
        return std::vector<FaceDetectionResult>();
    }
    
    return detectFaces(image);
}

std::vector<FaceDetectionResult> FaceDetector::detectFacesMultiScale(const cv::Mat& image) {
    if (!config_.enableMultiScale) {
        return detectFaces(image);
    }
    
    std::vector<FaceDetectionResult> allResults;
    std::vector<float> scales = {1.0f, 0.8f, 1.2f, 0.6f, 1.5f};
    
    for (float scale : scales) {
        cv::Size newSize(
            static_cast<int>(image.cols * scale),
            static_cast<int>(image.rows * scale)
        );
        
        cv::Mat resized = ImageUtils::resizeImage(image, newSize);
        auto results = detectFaces(resized);
        
        for (auto& result : results) {
            result.boundingBox.x = static_cast<int>(result.boundingBox.x / scale);
            result.boundingBox.y = static_cast<int>(result.boundingBox.y / scale);
            result.boundingBox.width = static_cast<int>(result.boundingBox.width / scale);
            result.boundingBox.height = static_cast<int>(result.boundingBox.height / scale);
            
            for (auto& pt : result.landmarks) {
                pt.x /= scale;
                pt.y /= scale;
            }
        }
        
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    for (const auto& result : allResults) {
        boxes.push_back(result.boundingBox);
        confidences.push_back(result.confidence);
    }
    
    auto indices = nonMaximumSuppression(boxes, confidences, config_.overlapThreshold);
    
    std::vector<FaceDetectionResult> finalResults;
    for (int idx : indices) {
        finalResults.push_back(allResults[idx]);
    }
    
    return finalResults;
}



// ============================================================================
// FACE COMPARISON & RECOGNITION
// ============================================================================

double FaceDetector::compareFaces(const cv::Mat& face1, const cv::Mat& face2) {
    if (face1.empty() || face2.empty()) {
        return 0.0;
    }
    
    try {
        // Method 1: Histogram comparison (fast and effective)
        double histScore = compareFacesHistogram(face1, face2);
        
        // Method 2: Structural similarity (SSIM)
        double ssimScore = compareFacesSSIM(face1, face2);
        
        // Method 3: Feature-based comparison (ORB/SIFT)
        double featureScore = compareFacesFeatures(face1, face2);
        
        // Method 4: Deep learning embedding (if available)
        double embeddingScore = compareFacesEmbeddings(face1, face2);
        
        // Combine scores with weights
        double finalScore = (histScore * 0.3) + (ssimScore * 0.3) + 
                           (featureScore * 0.2) + (embeddingScore * 0.2);
        
        return std::min(1.0, std::max(0.0, finalScore));
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Face comparison failed: " << e.what() << std::endl;
        return 0.0;
    }
}

double FaceDetector::compareFaces(const FaceDetectionResult& face1, const FaceDetectionResult& face2) {
    if (face1.faceImage.empty() && face2.faceImage.empty()) {
        return 0.0;
    }
    
    // Prefer aligned faces if available
    cv::Mat img1 = face1.alignedFace.empty() ? face1.faceImage : face1.alignedFace;
    cv::Mat img2 = face2.alignedFace.empty() ? face2.faceImage : face2.alignedFace;
    
    return compareFaces(img1, img2);
}

std::vector<std::pair<double, cv::Rect>> FaceDetector::findSimilarFaces(const cv::Mat& probeFace, const cv::Mat& image) {
    std::vector<std::pair<double, cv::Rect>> results;
    
    if (probeFace.empty() || image.empty()) {
        return results;
    }
    
    // Detect all faces in the image
    auto faces = detectFaces(image);
    
    for (const auto& face : faces) {
        if (face.faceImage.empty()) continue;
        
        double similarity = compareFaces(probeFace, face.faceImage);
        results.push_back({similarity, face.boundingBox});
    }
    
    // Sort by similarity score (descending)
    std::sort(results.begin(), results.end(),
        [](const std::pair<double, cv::Rect>& a, const std::pair<double, cv::Rect>& b) {
            return a.first > b.first;
        });
    
    return results;
}

std::vector<std::pair<double, std::string>> FaceDetector::findSimilarFacesInFolder(const cv::Mat& probeFace, const std::string& folderPath) {
    std::vector<std::pair<double, std::string>> results;
    
    if (probeFace.empty()) {
        return results;
    }
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
            if (!entry.is_regular_file()) continue;
            
            std::string filePath = entry.path().string();
            if (!ImageUtils::isValidImage(filePath)) continue;
            
            // Detect faces in the current image
            auto faces = detectFaces(filePath);
            
            double maxSimilarity = 0.0;
            for (const auto& face : faces) {
                if (face.faceImage.empty()) continue;
                
                double similarity = compareFaces(probeFace, face.faceImage);
                maxSimilarity = std::max(maxSimilarity, similarity);
            }
            
            if (maxSimilarity > 0.0) {
                results.push_back({maxSimilarity, filePath});
            }
        }
        
        // Sort by similarity score (descending)
        std::sort(results.begin(), results.end(),
            [](const std::pair<double, std::string>& a, const std::pair<double, std::string>& b) {
                return a.first > b.first;
            });
            
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Folder processing failed: " << e.what() << std::endl;
    }
    
    return results;
}

// ============================================================================
// FACE COMPARISON METHODS
// ============================================================================

double FaceDetector::compareFacesHistogram(const cv::Mat& face1, const cv::Mat& face2) {
    // Resize faces to same size for consistent comparison
    cv::Mat resized1, resized2;
    cv::Size targetSize(100, 100);
    
    cv::resize(face1, resized1, targetSize);
    cv::resize(face2, resized2, targetSize);
    
    // Convert to HSV color space for better color comparison
    cv::Mat hsv1, hsv2;
    cv::cvtColor(resized1, hsv1, cv::COLOR_BGR2HSV);
    cv::cvtColor(resized2, hsv2, cv::COLOR_BGR2HSV);
    
    // Calculate histograms
    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float* ranges[] = {h_ranges, s_ranges};
    int channels[] = {0, 1};
    
    cv::Mat hist1, hist2;
    cv::calcHist(&hsv1, 1, channels, cv::Mat(), hist1, 2, histSize, ranges, true, false);
    cv::calcHist(&hsv2, 1, channels, cv::Mat(), hist2, 2, histSize, ranges, true, false);
    
    // Normalize histograms
    cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    
    // Compare histograms using correlation (returns 1 for perfect match)
    double correlation = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    
    // Convert to 0-1 scale where 1 is perfect match
    return (correlation + 1.0) / 2.0;
}

double FaceDetector::compareFacesSSIM(const cv::Mat& face1, const cv::Mat& face2) {
    // Resize faces to same size
    cv::Mat resized1, resized2;
    cv::Size targetSize(100, 100);
    
    cv::resize(face1, resized1, targetSize);
    cv::resize(face2, resized2, targetSize);
    
    // Convert to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(resized1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resized2, gray2, cv::COLOR_BGR2GRAY);
    
    // Calculate SSIM (Structural Similarity Index)
    const double C1 = 6.5025, C2 = 58.5225;
    
    cv::Mat I1, I2;
    gray1.convertTo(I1, CV_32F);
    gray2.convertTo(I2, CV_32F);
    
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);
    
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    
    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    
    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    
    cv::Scalar mean_ssim = cv::mean(ssim_map);
    return mean_ssim[0];
}

double FaceDetector::compareFacesFeatures(const cv::Mat& face1, const cv::Mat& face2) {
    // Resize faces to same size
    cv::Mat resized1, resized2;
    cv::Size targetSize(200, 200);
    
    cv::resize(face1, resized1, targetSize);
    cv::resize(face2, resized2, targetSize);
    
    // Convert to grayscale
    cv::Mat gray1, gray2;
    cv::cvtColor(resized1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(resized2, gray2, cv::COLOR_BGR2GRAY);
    
    // Use ORB feature detector
    auto orb = cv::ORB::create(500);
    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    orb->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);
    
    if (descriptors1.empty() || descriptors2.empty()) {
        return 0.0;
    }
    
    // Use BFMatcher with Hamming distance
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);
    
    // Apply ratio test
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() < 2) continue;
        
        if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    // Calculate matching score
    double match_ratio = static_cast<double>(good_matches.size()) / 
                        std::min(keypoints1.size(), keypoints2.size());
    
    return std::min(1.0, match_ratio);
}

double FaceDetector::compareFacesEmbeddings(const cv::Mat& face1, const cv::Mat& face2) {
    // This method would use a pre-trained deep learning model
    // For now, we'll use a combination of other methods
    
    // Simple implementation using LBP features
    cv::Mat gray1, gray2;
    cv::cvtColor(face1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(face2, gray2, cv::COLOR_BGR2GRAY);
    
    // Resize for consistency
    cv::resize(gray1, gray1, cv::Size(100, 100));
    cv::resize(gray2, gray2, cv::Size(100, 100));
    
    // Calculate LBP features
    cv::Mat lbp1 = calculateLBP(gray1);
    cv::Mat lbp2 = calculateLBP(gray2);
    
    // Compare LBP histograms
    cv::Mat hist1, hist2;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    
    cv::calcHist(&lbp1, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
    cv::calcHist(&lbp2, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);
    
    cv::normalize(hist1, hist1, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(hist2, hist2, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    
    double correlation = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    return (correlation + 1.0) / 2.0;
}

cv::Mat FaceDetector::calculateLBP(const cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.rows-2, src.cols-2, CV_8UC1);
    
    for (int i = 1; i < src.rows-1; i++) {
        for (int j = 1; j < src.cols-1; j++) {
            uchar center = src.at<uchar>(i,j);
            unsigned char code = 0;
            code |= (src.at<uchar>(i-1,j-1) > center) << 7;
            code |= (src.at<uchar>(i-1,j) > center) << 6;
            code |= (src.at<uchar>(i-1,j+1) > center) << 5;
            code |= (src.at<uchar>(i,j+1) > center) << 4;
            code |= (src.at<uchar>(i+1,j+1) > center) << 3;
            code |= (src.at<uchar>(i+1,j) > center) << 2;
            code |= (src.at<uchar>(i+1,j-1) > center) << 1;
            code |= (src.at<uchar>(i,j-1) > center) << 0;
            dst.at<uchar>(i-1,j-1) = code;
        }
    }
    return dst;
}



// ============================================================================
// BATCH PROCESSING
// ============================================================================

std::vector<std::vector<FaceDetectionResult>> FaceDetector::detectFacesBatch(
    const std::vector<cv::Mat>& images,
    ProgressCallback callback) {
    
    std::vector<std::vector<FaceDetectionResult>> batchResults;
    batchResults.reserve(images.size());
    
    for (size_t i = 0; i < images.size(); ++i) {
        auto results = detectFaces(images[i]);
        batchResults.push_back(results);
        
        if (callback) {
            callback(i + 1, images.size(), "Processing image " + std::to_string(i + 1));
        }
    }
    
    return batchResults;
}

std::vector<std::vector<FaceDetectionResult>> FaceDetector::detectFacesBatch(
    const std::vector<std::string>& imagePaths,
    ProgressCallback callback) {
    
    std::vector<std::vector<FaceDetectionResult>> batchResults;
    batchResults.reserve(imagePaths.size());
    
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        auto results = detectFaces(imagePaths[i]);
        batchResults.push_back(results);
        
        if (callback) {
            callback(i + 1, imagePaths.size(), "Processing: " + imagePaths[i]);
        }
    }
    
    return batchResults;
}

// ============================================================================
// INDIVIDUAL DETECTION METHODS
// ============================================================================

std::vector<cv::Rect> FaceDetector::detectFacesDlib(const cv::Mat& image) {
    if (!dlibDetector_) {
        return std::vector<cv::Rect>();
    }
    
    try {
        cv::Mat gray = ImageUtils::convertToGray(image);
        dlib::cv_image<unsigned char> dlibImage(gray);
        
        std::vector<dlib::rectangle> dlibFaces = (*dlibDetector_)(dlibImage);
        
        std::vector<cv::Rect> faces;
        faces.reserve(dlibFaces.size());
        
        for (const auto& face : dlibFaces) {
            cv::Rect rect(face.left(), face.top(), face.width(), face.height());
            
            if (rect.width >= config_.minFaceSize.width && 
                rect.height >= config_.minFaceSize.height) {
                faces.push_back(rect);
            }
        }
        
        return faces;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Dlib detection failed: " << e.what() << std::endl;
        return std::vector<cv::Rect>();
    }
}

std::vector<cv::Rect> FaceDetector::detectFacesHaar(const cv::Mat& image) {
    try {
        if (haarCascade_.empty()) {
            std::string cascadePath = cv::samples::findFile("/home/cpluspluser/Projects/FaceOrganizer/model/haarcascade_frontalface_default.xml");
            if (!haarCascade_.load(cascadePath)) {
                std::cerr << "[ERROR] Failed to load Haar cascade" << std::endl;
                return std::vector<cv::Rect>();
            }
        }
        
        cv::Mat gray = ImageUtils::convertToGray(image);
        std::vector<cv::Rect> faces;
        
        haarCascade_.detectMultiScale(
            gray,
            faces,
            config_.scaleFactor,
            config_.minNeighbors,
            0,
            config_.minFaceSize,
            config_.maxFaceSize
        );
        
        return faces;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] Haar detection failed: " << e.what() << std::endl;
        return std::vector<cv::Rect>();
    }
}

std::vector<cv::Rect> FaceDetector::detectFacesLBP(const cv::Mat& image) {
    try {
        if (lbpCascade_.empty()) {
            std::string cascadePath = cv::samples::findFile("/home/cpluspluser/Projects/FaceOrganizer/model/lbpcascade_frontalface.xml");
            if (!lbpCascade_.load(cascadePath)) {
                std::cerr << "[ERROR] Failed to load LBP cascade" << std::endl;
                return std::vector<cv::Rect>();
            }
        }
        
        cv::Mat gray = ImageUtils::convertToGray(image);
        std::vector<cv::Rect> faces;
        
        lbpCascade_.detectMultiScale(
            gray,
            faces,
            config_.scaleFactor,
            config_.minNeighbors,
            0,
            config_.minFaceSize,
            config_.maxFaceSize
        );
        
        return faces;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] LBP detection failed: " << e.what() << std::endl;
        return std::vector<cv::Rect>();
    }
}

std::vector<cv::Rect> FaceDetector::detectFacesDNN(const cv::Mat& image) {
    if (dnnNet_.empty()) {
        return std::vector<cv::Rect>();
    }
    
    try {
        const int INPUT_SIZE = 300;
        cv::Mat blob = cv::dnn::blobFromImage(
            image,
            1.0,
            cv::Size(INPUT_SIZE, INPUT_SIZE),
            cv::Scalar(104.0, 177.0, 123.0),
            false,
            false
        );
        
        dnnNet_.setInput(blob);
        cv::Mat detection = dnnNet_.forward();
        
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
        
        std::vector<cv::Rect> faces;
        
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);
            
            if (confidence > config_.minConfidence) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * image.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * image.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * image.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * image.rows);
                
                cv::Rect rect(x1, y1, x2 - x1, y2 - y1);
                
                if (rect.width >= config_.minFaceSize.width && 
                    rect.height >= config_.minFaceSize.height) {
                    faces.push_back(rect);
                }
            }
        }
        
        return faces;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] DNN detection failed: " << e.what() << std::endl;
        return std::vector<cv::Rect>();
    }
}

std::vector<cv::Rect> FaceDetector::detectFacesYuNet(const cv::Mat& image) {
    if (!yunetDetector_) {
        return std::vector<cv::Rect>();
    }
    
    try {
        yunetDetector_->setInputSize(image.size());
        
        cv::Mat faces;
        yunetDetector_->detect(image, faces);
        
        std::vector<cv::Rect> faceRects;
        
        for (int i = 0; i < faces.rows; i++) {
            float confidence = faces.at<float>(i, 14);
            
            if (confidence > config_.minConfidence) {
                int x = static_cast<int>(faces.at<float>(i, 0));
                int y = static_cast<int>(faces.at<float>(i, 1));
                int w = static_cast<int>(faces.at<float>(i, 2));
                int h = static_cast<int>(faces.at<float>(i, 3));
                
                cv::Rect rect(x, y, w, h);
                
                if (rect.width >= config_.minFaceSize.width && 
                    rect.height >= config_.minFaceSize.height) {
                    faceRects.push_back(rect);
                }
            }
        }
        
        return faceRects;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] YuNet detection failed: " << e.what() << std::endl;
        return std::vector<cv::Rect>();
    }
}

// ============================================================================
// LANDMARK DETECTION
// ============================================================================

std::vector<cv::Point2f> FaceDetector::detectLandmarks(
    const cv::Mat& faceImage, 
    LandmarkMethod method) {
    
    if (faceImage.empty()) {
        return std::vector<cv::Point2f>();
    }
    
    switch (method) {
        case LandmarkMethod::DLIB_68:
            return detectLandmarksDlib(faceImage);
        case LandmarkMethod::LBF:
            return detectLandmarksLBF(faceImage);
        case LandmarkMethod::YUNET:
            return detectLandmarksYuNet(faceImage);
        case LandmarkMethod::HAAR_BASED:
            return detectLandmarksHaar(faceImage);
        case LandmarkMethod::GENERATED:
            return generateLandmarks(faceImage);
        default:
            return generateLandmarks(faceImage);
    }
}

std::vector<cv::Point2f> FaceDetector::detectLandmarksDlib(const cv::Mat& faceImage) 
{
    return ImageUtils::detectLandmarksDlib(faceImage);
}

std::vector<cv::Point2f> FaceDetector::detectLandmarksLBF(const cv::Mat& faceImage) 
{
    return ImageUtils::detectLandmarksLBF(faceImage);
}

std::vector<cv::Point2f> FaceDetector::detectLandmarksYuNet(const cv::Mat& faceImage) {
    return ImageUtils::detectFacialLandmarksYuNet(faceImage);
}

std::vector<cv::Point2f> FaceDetector::detectLandmarksHaar(const cv::Mat& faceImage) {
    return ImageUtils::detectLandmarksHAAR(faceImage);
}

std::vector<cv::Point2f> FaceDetector::generateLandmarks(const cv::Mat& faceImage) {
    return ImageUtils::generateDefaultLandmarks(faceImage);
}

// ============================================================================
// FACE ALIGNMENT
// ============================================================================

cv::Mat FaceDetector::alignFace(
    const cv::Mat& image, 
    const std::vector<cv::Point2f>& landmarks) {
    return alignFace(image, landmarks, config_.alignmentMethod);
}

cv::Mat FaceDetector::alignFace(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks,
    AlignmentMethod method) {
    
    switch (method) {
        case AlignmentMethod::SIMILARITY:
            return alignFaceSimilarity(image, landmarks);
        case AlignmentMethod::AFFINE:
            return alignFaceAffine(image, landmarks);
        case AlignmentMethod::PERSPECTIVE:
            return alignFacePerspective(image, landmarks);
        case AlignmentMethod::EYES_CENTER:
            return alignFaceEyesCenter(image, landmarks);
        default:
            return alignFaceSimilarity(image, landmarks);
    }
}

cv::Mat FaceDetector::alignFaceSimilarity(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks) {
    
    if (landmarks.size() < 68) {
        return cv::Mat();
    }
    
    cv::Point2f leftEye = (landmarks[36] + landmarks[39]) * 0.5f;
    cv::Point2f rightEye = (landmarks[42] + landmarks[45]) * 0.5f;
    
    float dy = rightEye.y - leftEye.y;
    float dx = rightEye.x - leftEye.x;
    float angle = std::atan2(dy, dx) * 180.0 / CV_PI;
    
    cv::Point2f desiredLeftEye(config_.alignedFaceSize.width * 0.35f, 
                               config_.alignedFaceSize.height * 0.35f);
    cv::Point2f desiredRightEye(config_.alignedFaceSize.width * 0.65f,
                                config_.alignedFaceSize.height * 0.35f);
    
    float desiredDist = cv::norm(desiredRightEye - desiredLeftEye);
    float actualDist = cv::norm(rightEye - leftEye);
    float scale = desiredDist / actualDist;
    
    cv::Point2f eyesCenter = (leftEye + rightEye) * 0.5f;
    
    cv::Mat M = cv::getRotationMatrix2D(eyesCenter, angle, scale);
    
    M.at<double>(0, 2) += (config_.alignedFaceSize.width * 0.5 - eyesCenter.x);
    M.at<double>(1, 2) += (config_.alignedFaceSize.height * 0.5 - eyesCenter.y);
    
    cv::Mat aligned;
    cv::warpAffine(image, aligned, M, config_.alignedFaceSize, cv::INTER_LINEAR);
    
    return aligned;
}

cv::Mat FaceDetector::alignFaceAffine(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks) {
    
    if (landmarks.size() < 68) {
        return cv::Mat();
    }
    
    std::vector<cv::Point2f> srcPoints = {
        (landmarks[36] + landmarks[39]) * 0.5f,
        (landmarks[42] + landmarks[45]) * 0.5f,
        landmarks[30]
    };
    
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(config_.alignedFaceSize.width * 0.35f, config_.alignedFaceSize.height * 0.35f),
        cv::Point2f(config_.alignedFaceSize.width * 0.65f, config_.alignedFaceSize.height * 0.35f),
        cv::Point2f(config_.alignedFaceSize.width * 0.50f, config_.alignedFaceSize.height * 0.65f)
    };
    
    cv::Mat M = cv::getAffineTransform(srcPoints, dstPoints);
    
    cv::Mat aligned;
    cv::warpAffine(image, aligned, M, config_.alignedFaceSize, cv::INTER_LINEAR);
    
    return aligned;
}

cv::Mat FaceDetector::alignFacePerspective(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks) {
    
    if (landmarks.size() < 68) {
        return cv::Mat();
    }
    
    std::vector<cv::Point2f> srcPoints = {
        (landmarks[36] + landmarks[39]) * 0.5f,
        (landmarks[42] + landmarks[45]) * 0.5f,
        landmarks[48],
        landmarks[54]
    };
    
    std::vector<cv::Point2f> dstPoints = {
        cv::Point2f(config_.alignedFaceSize.width * 0.35f, config_.alignedFaceSize.height * 0.35f),
        cv::Point2f(config_.alignedFaceSize.width * 0.65f, config_.alignedFaceSize.height * 0.35f),
        cv::Point2f(config_.alignedFaceSize.width * 0.35f, config_.alignedFaceSize.height * 0.75f),
        cv::Point2f(config_.alignedFaceSize.width * 0.65f, config_.alignedFaceSize.height * 0.75f)
    };
    
    cv::Mat M = cv::getPerspectiveTransform(srcPoints, dstPoints);
    
    cv::Mat aligned;
    cv::warpPerspective(image, aligned, M, config_.alignedFaceSize, cv::INTER_LINEAR);
    
    return aligned;
}

cv::Mat FaceDetector::alignFaceEyesCenter(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks) {
    
    if (landmarks.size() < 68) {
        return cv::Mat();
    }
    
    cv::Point2f leftEye = (landmarks[36] + landmarks[39]) * 0.5f;
    cv::Point2f rightEye = (landmarks[42] + landmarks[45]) * 0.5f;
    
    float dy = rightEye.y - leftEye.y;
    float dx = rightEye.x - leftEye.x;
    float angle = std::atan2(dy, dx) * 180.0 / CV_PI;
    
    cv::Point2f center = (leftEye + rightEye) * 0.5f;
    
    cv::Mat rotated = ImageUtils::rotateImage(image, angle);
    
    int x = std::max(0, static_cast<int>(center.x - config_.alignedFaceSize.width / 2));
    int y = std::max(0, static_cast<int>(center.y - config_.alignedFaceSize.height / 2));
    int w = std::min(config_.alignedFaceSize.width, rotated.cols - x);
    int h = std::min(config_.alignedFaceSize.height, rotated.rows - y);
    
    cv::Rect roi(x, y, w, h);
    cv::Mat aligned = ImageUtils::cropImageSafe(rotated, roi);
    
    if (aligned.size() != config_.alignedFaceSize) {
        aligned = ImageUtils::resizeImage(aligned, config_.alignedFaceSize);
    }
    
    return aligned;
}

// ============================================================================
// FACE EXTRACTION & PROCESSING
// ============================================================================

// cv::Mat FaceDetector::extractFace(
//     const cv::Mat& image,
//     const cv::Rect& faceRect,
//     bool addPadding) {
    
//     return ImageUtils::extractFace(
//         image,
//         faceRect,
//         addPadding,
//         config_.facePadding
//     );
// }

std::vector<cv::Mat> FaceDetector::extractAllFaces(const cv::Mat& image) {
    auto results = detectFaces(image);
    
    std::vector<cv::Mat> faces;
    faces.reserve(results.size());
    
    for (const auto& result : results) {
        if (!result.faceImage.empty()) {
            faces.push_back(result.faceImage);
        }
    }
    
    return faces;
}

// ============================================================================
// POST-PROCESSING & FILTERING
// ============================================================================

std::vector<cv::Rect> FaceDetector::filterOverlappingFaces(
    const std::vector<cv::Rect>& faces,
    const std::vector<float>& confidences) {
    
    if (faces.empty()) {
        return faces;
    }
    
    auto indices = nonMaximumSuppression(faces, confidences, config_.overlapThreshold);
    
    std::vector<cv::Rect> filtered;
    filtered.reserve(indices.size());
    
    for (int idx : indices) {
        filtered.push_back(faces[idx]);
    }
    
    return filtered;
}

float FaceDetector::calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);
    
    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = rect1.area() + rect2.area() - intersectionArea;
    
    return unionArea > 0 ? static_cast<float>(intersectionArea) / unionArea : 0.0f;
}

std::vector<int> FaceDetector::nonMaximumSuppression(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float threshold) {
    
    if (boxes.empty()) {
        return std::vector<int>();
    }
    
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
        [&scores](int i1, int i2) {
            return scores[i1] > scores[i2];
        });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        
        if (suppressed[idx]) {
            continue;
        }
        
        keep.push_back(idx);
        
        for (size_t j = i + 1; j < indices.size(); ++j) {
            int idx2 = indices[j];
            
            if (suppressed[idx2]) {
                continue;
            }
            
            float iou = calculateIOU(boxes[idx], boxes[idx2]);
            
            if (iou > threshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}

// ============================================================================
// FACE QUALITY ASSESSMENT
// ============================================================================

float FaceDetector::assessFaceQuality(const cv::Mat& faceImage) {
    if (faceImage.empty()) {
        return 0.0f;
    }
    
    float sharpness = calculateSharpness(faceImage);
    float brightness = calculateBrightness(faceImage);
    float contrast = calculateContrast(faceImage);
    
    sharpness = std::min(1.0f, sharpness / 100.0f);
    brightness = 1.0f - std::abs(brightness - 0.5f) * 2.0f;
    contrast = std::min(1.0f, contrast / 50.0f);
    
    float quality = 0.5f * sharpness + 0.3f * contrast + 0.2f * brightness;
    
    return quality;
}

float FaceDetector::calculateSharpness(const cv::Mat& image) {
    cv::Mat gray = ImageUtils::convertToGray(image);
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    return static_cast<float>(stddev[0] * stddev[0]);
}

float FaceDetector::calculateBrightness(const cv::Mat& image) {
    cv::Mat gray = ImageUtils::convertToGray(image);
    cv::Scalar meanValue = cv::mean(gray);
    
    return static_cast<float>(meanValue[0] / 255.0);
}

float FaceDetector::calculateContrast(const cv::Mat& image) {
    cv::Mat gray = ImageUtils::convertToGray(image);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    return static_cast<float>(stddev[0]);
}

bool FaceDetector::isFaceFrontal(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() < 68) {
        return false;
    }
    
    float leftEyeWidth = cv::norm(landmarks[39] - landmarks[36]);
    float rightEyeWidth = cv::norm(landmarks[45] - landmarks[42]);
    float eyeSymmetry = std::min(leftEyeWidth, rightEyeWidth) / 
                        std::max(leftEyeWidth, rightEyeWidth);
    
    float topWidth = cv::norm(landmarks[16] - landmarks[0]);
    float midWidth = cv::norm(landmarks[14] - landmarks[2]);
    float widthRatio = std::min(topWidth, midWidth) / std::max(topWidth, midWidth);
    
    return (eyeSymmetry > 0.85f && widthRatio > 0.80f);
}

// ============================================================================
// VISUALIZATION
// ============================================================================

cv::Mat FaceDetector::drawFaceDetections(
    const cv::Mat& image,
    const std::vector<FaceDetectionResult>& results,
    bool showLandmarks,
    bool showConfidence) {
    
    cv::Mat output = image.clone();
    
    for (const auto& result : results) {
        cv::Scalar color(0, 255, 0);
        if (result.confidence < 0.7f) {
            color = cv::Scalar(0, 165, 255);
        }
        
        cv::rectangle(output, result.boundingBox, color, 2);
        
        if (showConfidence) {
            std::string confidenceText = "Conf: " + 
                std::to_string(static_cast<int>(result.confidence * 100)) + "%";
            
            cv::putText(output, confidenceText,
                       cv::Point(result.boundingBox.x, result.boundingBox.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
        }
        
        if (showLandmarks && !result.landmarks.empty()) {
            for (const auto& pt : result.landmarks) {
                cv::circle(output, pt, 2, cv::Scalar(255, 0, 0), -1);
            }
        }
    }
    
    return output;
}

cv::Mat FaceDetector::drawBoundingBoxes(
    const cv::Mat& image,
    const std::vector<cv::Rect>& faces) {
    
    cv::Mat output = image.clone();
    
    for (const auto& face : faces) {
        cv::rectangle(output, face, cv::Scalar(0, 255, 0), 2);
    }
    
    return output;
}

cv::Mat FaceDetector::drawLandmarks(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& landmarks,
    bool withNumbers) {
    
    cv::Mat output = image.clone();
    
    cv::Scalar jawColor(255, 0, 0);
    cv::Scalar eyebrowColor(0, 255, 0);
    cv::Scalar noseColor(0, 0, 255);
    cv::Scalar eyeColor(255, 255, 0);
    cv::Scalar mouthColor(255, 0, 255);
    
    for (size_t i = 0; i < landmarks.size(); ++i) {
        cv::Scalar color;
        
        if (i < 17) color = jawColor;
        else if (i < 27) color = eyebrowColor;
        else if (i < 36) color = noseColor;
        else if (i < 48) color = eyeColor;
        else color = mouthColor;
        
        cv::circle(output, landmarks[i], 2, color, -1);
        
        if (withNumbers) {
            cv::putText(output, std::to_string(i), landmarks[i],
                       cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
        }
    }
    
    if (landmarks.size() >= 68) {
        for (int i = 0; i < 16; ++i) {
            cv::line(output, landmarks[i], landmarks[i + 1], jawColor, 1);
        }
        
        for (int i = 17; i < 21; ++i) {
            cv::line(output, landmarks[i], landmarks[i + 1], eyebrowColor, 1);
        }
        for (int i = 22; i < 26; ++i) {
            cv::line(output, landmarks[i], landmarks[i + 1], eyebrowColor, 1);
        }
        
        for (int i = 36; i < 41; ++i) {
            cv::line(output, landmarks[i], landmarks[i + 1], eyeColor, 1);
        }
        cv::line(output, landmarks[41], landmarks[36], eyeColor, 1);
        
        for (int i = 42; i < 47; ++i) {
            cv::line(output, landmarks[i], landmarks[i + 1], eyeColor, 1);
        }
        cv::line(output, landmarks[47], landmarks[42], eyeColor, 1);
        
        for (int i = 48; i < 59; ++i) {
            cv::line(output, landmarks[i], landmarks[i + 1], mouthColor, 1);
        }
        cv::line(output, landmarks[59], landmarks[48], mouthColor, 1);
    }
    
    return output;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

void FaceDetector::preprocessImage(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) {
        output = cv::Mat();
        return;
    }
    
    if (input.channels() == 1) {
        cv::cvtColor(input, output, cv::COLOR_GRAY2BGR);
    } else {
        output = input.clone();
    }
    
    if (config_.detectionMethod == DetectionMethod::HAAR_CASCADE ||
        config_.detectionMethod == DetectionMethod::LBP_CASCADE) {
        output = ImageUtils::equalizeHistogram(output);
    }
}

cv::Mat FaceDetector::extractFace(
    const cv::Mat& image,
    const cv::Rect& faceRect,
    bool addPadding) {
    
    if (image.empty() || faceRect.area() == 0) {
        return cv::Mat();
    }

    // تأكد أن الـ faceRect داخل حدود الصورة أصلاً
    cv::Rect safeFaceRect = getSafeFaceRect(faceRect, image.size());
    if (safeFaceRect.area() == 0) {
        return cv::Mat();
    }

    cv::Rect expandedRect = safeFaceRect;
    
    if (addPadding) {
        int padX = static_cast<int>(safeFaceRect.width * config_.facePadding);
        int padY = static_cast<int>(safeFaceRect.height * config_.facePadding);
        
        expandedRect.x -= padX;
        expandedRect.y -= padY;
        expandedRect.width += 2 * padX;
        expandedRect.height += 2 * padY;
        
        // تأكد أن المنطقة الموسعة داخل الحدود
        expandedRect = getSafeFaceRect(expandedRect, image.size());
    }

    // إذا المنطقة أصبحت فارغة، ارجع صورة فارغة
    if (expandedRect.width <= 0 || expandedRect.height <= 0) {
        return cv::Mat();
    }

    return image(expandedRect).clone();
}

// أضف هذه الدالة المساعدة
cv::Rect FaceDetector::getSafeFaceRect(const cv::Rect& rect, const cv::Size& imageSize) {
    int x = std::max(0, rect.x);
    int y = std::max(0, rect.y);
    int width = std::min(rect.width, imageSize.width - x);
    int height = std::min(rect.height, imageSize.height - y);
    
    return (width > 0 && height > 0) ? cv::Rect(x, y, width, height) : cv::Rect(0, 0, 0, 0);
}

cv::Rect FaceDetector::expandFaceRect(
    const cv::Rect& rect,
    const cv::Size& imageSize,
    float padding) {
    
    int padX = static_cast<int>(rect.width * padding);
    int padY = static_cast<int>(rect.height * padding);
    
    cv::Rect expanded(
        std::max(0, rect.x - padX),
        std::max(0, rect.y - padY),
        std::min(rect.width + 2 * padX, imageSize.width - rect.x + padX),
        std::min(rect.height + 2 * padY, imageSize.height - rect.y + padY)
    );
    
    return expanded;
}

std::vector<cv::Point2f> FaceDetector::getEyePositions(
    const std::vector<cv::Point2f>& landmarks) {
    
    if (landmarks.size() < 68) {
        return std::vector<cv::Point2f>();
    }
    
    cv::Point2f leftEye = (landmarks[36] + landmarks[39]) * 0.5f;
    cv::Point2f rightEye = (landmarks[42] + landmarks[45]) * 0.5f;
    
    return {leftEye, rightEye};
}

cv::Mat FaceDetector::getRotationMatrix(
    const cv::Point2f& leftEye,
    const cv::Point2f& rightEye) {
    
    float dy = rightEye.y - leftEye.y;
    float dx = rightEye.x - leftEye.x;
    float angle = std::atan2(dy, dx) * 180.0 / CV_PI;
    
    cv::Point2f center = (leftEye + rightEye) * 0.5f;
    
    return cv::getRotationMatrix2D(center, angle, 1.0);
}

std::vector<cv::Point2f> FaceDetector::transformLandmarks(
    const std::vector<cv::Point2f>& landmarks,
    const cv::Mat& transformMatrix) {
    
    std::vector<cv::Point2f> transformed;
    transformed.reserve(landmarks.size());
    
    for (const auto& pt : landmarks) {
        float x = transformMatrix.at<double>(0, 0) * pt.x + 
                  transformMatrix.at<double>(0, 1) * pt.y + 
                  transformMatrix.at<double>(0, 2);
        float y = transformMatrix.at<double>(1, 0) * pt.x + 
                  transformMatrix.at<double>(1, 1) * pt.y + 
                  transformMatrix.at<double>(1, 2);
        
        transformed.push_back(cv::Point2f(x, y));
    }
    
    return transformed;
}

void FaceDetector::sortResultsByConfidence(std::vector<FaceDetectionResult>& results) {
    std::sort(results.begin(), results.end(),
        [](const FaceDetectionResult& a, const FaceDetectionResult& b) {
            return a.confidence > b.confidence;
        });
}

float FaceDetector::calculateDetectionConfidence(
    const cv::Rect& face,
    const cv::Mat& image) {
    
    cv::Mat faceRegion = image(face);
    
    float quality = assessFaceQuality(faceRegion);
    float sizeScore = std::min(1.0f, face.area() / 10000.0f);
    
    return (quality + sizeScore) * 0.5f;
}

std::vector<cv::Point2f> FaceDetector::generateLandmarksFrom5Points(
    const std::vector<cv::Point2f>& keyPoints,
    const cv::Size& faceSize) {
    
    return ImageUtils::generateLandmarksFromKeyPoints(keyPoints, faceSize);
}

// ============================================================================
// STATISTICS & ERROR HANDLING
// ============================================================================

int FaceDetector::getTotalDetections() const {
    return totalDetections_;
}

double FaceDetector::getAverageConfidence() const {
    return totalDetections_ > 0 ? totalConfidence_ / totalDetections_ : 0.0;
}

std::string FaceDetector::getDetectionMethodName(DetectionMethod method) const {
    switch (method) {
        case DetectionMethod::DLIB: return "Dlib HOG";
        case DetectionMethod::HAAR_CASCADE: return "Haar Cascade";
        case DetectionMethod::LBP_CASCADE: return "LBP Cascade";
        case DetectionMethod::DNN_CAFFE: return "DNN Caffe";
        case DetectionMethod::DNN_TENSORFLOW: return "DNN TensorFlow";
        case DetectionMethod::YUNET: return "YuNet";
        case DetectionMethod::AUTO: return "Auto";
        default: return "Unknown";
    }
}

std::string FaceDetector::getLandmarkMethodName(LandmarkMethod method) const {
    switch (method) {
        case LandmarkMethod::DLIB_68: return "Dlib 68-point";
        case LandmarkMethod::LBF: return "LBF";
        case LandmarkMethod::YUNET: return "YuNet";
        case LandmarkMethod::HAAR_BASED: return "Haar-based";
        case LandmarkMethod::GENERATED: return "Generated";
        default: return "Unknown";
    }
}

void FaceDetector::resetStatistics() {
    totalDetections_ = 0;
    totalConfidence_ = 0.0;
}

std::string FaceDetector::getLastError() const {
    return lastError_;
}

bool FaceDetector::hasError() const {
    return !lastError_.empty();
}

// ============================================================================
// HELPER FUNCTIONS (FaceDetectionUtils namespace)
// ============================================================================

namespace FaceDetectionUtils {

cv::Rect dlibRectToCV(const dlib::rectangle& rect) {
    return cv::Rect(rect.left(), rect.top(), rect.width(), rect.height());
}

dlib::rectangle cvRectToDlib(const cv::Rect& rect) {
    return dlib::rectangle(rect.x, rect.y, rect.x + rect.width, rect.y + rect.height);
}

std::vector<cv::Point2f> dlibPointsToCV(const dlib::full_object_detection& shape) {
    std::vector<cv::Point2f> points;
    points.reserve(shape.num_parts());
    
    for (unsigned int i = 0; i < shape.num_parts(); ++i) {
        points.push_back(cv::Point2f(shape.part(i).x(), shape.part(i).y()));
    }
    
    return points;
}

bool areSameFace(
    const FaceDetectionResult& face1,
    const FaceDetectionResult& face2,
    float threshold) {
    
    float similarity = calculateFaceSimilarity(face1, face2);
    return similarity > threshold;
}

float calculateFaceSimilarity(
    const FaceDetectionResult& face1,
    const FaceDetectionResult& face2) {
    
    int x1 = std::max(face1.boundingBox.x, face2.boundingBox.x);
    int y1 = std::max(face1.boundingBox.y, face2.boundingBox.y);
    int x2 = std::min(face1.boundingBox.x + face1.boundingBox.width,
                      face2.boundingBox.x + face2.boundingBox.width);
    int y2 = std::min(face1.boundingBox.y + face1.boundingBox.height,
                      face2.boundingBox.y + face2.boundingBox.height);
    
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int union_area = face1.boundingBox.area() + face2.boundingBox.area() - intersection;
    
    return union_area > 0 ? static_cast<float>(intersection) / union_area : 0.0f;
}

cv::Mat preprocessForDetection(const cv::Mat& image) {
    return ImageUtils::equalizeHistogram(image);
}

cv::Mat enhanceContrast(const cv::Mat& image) {
    return ImageUtils::adjustBrightnessContrast(image, 1.2, 10);
}

bool isValidFaceRect(const cv::Rect& rect, const cv::Size& imageSize) {
    return rect.x >= 0 && rect.y >= 0 &&
           rect.x + rect.width <= imageSize.width &&
           rect.y + rect.height <= imageSize.height &&
           rect.width > 0 && rect.height > 0;
}

bool hasValidLandmarks(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.empty()) {
        return false;
    }
    
    for (const auto& pt : landmarks) {
        if (pt.x < 0 || pt.y < 0 || std::isnan(pt.x) || std::isnan(pt.y)) {
            return false;
        }
    }
    
    return true;
}

float estimateYaw(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() < 68) {
        return 0.0f;
    }
    
    float leftDist = cv::norm(landmarks[30] - landmarks[0]);
    float rightDist = cv::norm(landmarks[30] - landmarks[16]);
    
    float yaw = (leftDist - rightDist) / (leftDist + rightDist);
    
    return yaw * 90.0f;
}

float estimatePitch(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() < 68) {
        return 0.0f;
    }
    
    float eyeY = (landmarks[36].y + landmarks[45].y) * 0.5f;
    float noseY = landmarks[30].y;
    float mouthY = landmarks[51].y;
    
    float eyeNoseDist = noseY - eyeY;
    float noseMouthDist = mouthY - noseY;
    
    float pitch = (eyeNoseDist - noseMouthDist) / (eyeNoseDist + noseMouthDist);
    
    return pitch * 90.0f;
}

float estimateRoll(const std::vector<cv::Point2f>& landmarks) {
    if (landmarks.size() < 68) {
        return 0.0f;
    }
    
    cv::Point2f leftEye = (landmarks[36] + landmarks[39]) * 0.5f;
    cv::Point2f rightEye = (landmarks[42] + landmarks[45]) * 0.5f;
    
    float dy = rightEye.y - leftEye.y;
    float dx = rightEye.x - leftEye.x;
    
    return std::atan2(dy, dx) * 180.0f / CV_PI;
}

} // namespace FaceDetectionUtils
