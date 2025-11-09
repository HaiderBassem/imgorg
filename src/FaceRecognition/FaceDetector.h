#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <mutex>  // أضفنا هذا

// Face detection result structure
struct FaceDetectionResult {
    cv::Rect boundingBox;                    // Face bounding box in image
    std::vector<cv::Point2f> landmarks;      // 68-point facial landmarks
    float confidence;                         // Detection confidence score (0.0 - 1.0)
    cv::Mat faceImage;                       // Extracted face region
    cv::Mat alignedFace;                     // Aligned face (normalized orientation)
    int detectionMethod;                     // Which method detected this face
    
    FaceDetectionResult() : confidence(0.0f), detectionMethod(0) {}
};

// Detection method flags
enum class DetectionMethod {
    DLIB = 0,           // Dlib HOG-based detector
    HAAR_CASCADE = 1,   // OpenCV Haar Cascade
    LBP_CASCADE = 2,    // OpenCV LBP Cascade
    DNN_CAFFE = 3,      // OpenCV DNN with Caffe model
    DNN_TENSORFLOW = 4, // OpenCV DNN with TensorFlow model
    YUNET = 5,          // YuNet face detector
    AUTO = 6            // Automatically select best method
};

// Landmark detection method
enum class LandmarkMethod {
    DLIB_68 = 0,        // Dlib 68-point predictor
    LBF = 1,            // Local Binary Features
    YUNET = 2,          // YuNet 5-point landmarks + enhancement
    HAAR_BASED = 3,     // HAAR-based estimation
    GENERATED = 4       // Generated from face proportions
};

// Face alignment method
enum class AlignmentMethod {
    SIMILARITY = 0,     // Similarity transform (rotation + scale)
    AFFINE = 1,         // Affine transform (6 DOF)
    PERSPECTIVE = 2,    // Perspective transform (8 DOF)
    EYES_CENTER = 3     // Simple eye-based centering
};

// Configuration for face detection
struct FaceDetectorConfig {
    // Detection settings
    DetectionMethod detectionMethod = DetectionMethod::AUTO;
    float minConfidence = 0.5f;                    // Minimum confidence threshold
    cv::Size minFaceSize = cv::Size(80, 80);       // Minimum face size to detect
    cv::Size maxFaceSize = cv::Size(0, 0);         // Maximum face size (0 = no limit)
    float scaleFactor = 1.1f;                      // Multi-scale detection scale factor
    int minNeighbors = 3;                          // Min neighbors for cascade methods
    
    // Landmark settings
    LandmarkMethod landmarkMethod = LandmarkMethod::DLIB_68;
    bool detectLandmarks = true;                   // Enable landmark detection
    bool enhanceLandmarks = true;                  // Enhance landmarks with additional points
    
    // Face extraction settings
    bool extractFaces = true;                      // Extract face regions
    bool alignFaces = true;                        // Align extracted faces
    AlignmentMethod alignmentMethod = AlignmentMethod::SIMILARITY;
    cv::Size alignedFaceSize = cv::Size(160, 160); // Size for aligned faces
    float facePadding = 0.2f;                      // Padding around face (0.0 - 1.0)
    
    // Performance settings
    bool useGPU = false;                           // Use GPU acceleration if available
    bool enableMultiScale = true;                  // Enable multi-scale detection
    int numThreads = 4;                            // Number of threads for processing
    
    // Model paths
    std::string dlibModelPath  = "/home/cpluspluser/Projects/FaceOrganizer/model/shape_predictor_68_face_landmarks.dat";
    std::string lbfModelPath   = "/home/cpluspluser/Projects/FaceOrganizer/model/lbfmodel.yaml";
    std::string yunetModelPath = "/home/cpluspluser/Projects/FaceOrganizer/model/face_detection_yunet_2023mar.onnx";
    std::string caffeProtoPath = "/home/cpluspluser/Projects/FaceOrganizer/model/deploy.prototxt";
    std::string caffeModelPath = "/home/cpluspluser/Projects/FaceOrganizer/model/res10_300x300_ssd_iter_140000.caffemodel";
    std::string tfModelPath    = "/home/cpluspluser/Projects/FaceOrganizer/model/opencv_face_detector_uint8.pb";
    std::string tfConfigPath   = "/home/cpluspluser/Projects/FaceOrganizer/model/opencv_face_detector.pbtxt";
    std::string haarCascade    = "/home/cpluspluser/Projects/FaceOrganizer/model/haarcascade_frontalface_default.xml";

    
    // Advanced settings
    bool filterOverlappingFaces = true;            // Remove overlapping detections
    float overlapThreshold = 0.3f;                 // IOU threshold for overlap filtering
    bool sortByConfidence = true;                  // Sort results by confidence
    int maxFacesPerImage = 100;                    // Maximum faces to detect per image
};

// Progress callback type
using ProgressCallback = std::function<void(int current, int total, const std::string& message)>;

class FaceDetector {
public:
    // Constructor & Destructor
    FaceDetector();
    explicit FaceDetector(const FaceDetectorConfig& config);
    ~FaceDetector();
    
    // Initialization
    bool initialize();
    bool initialize(const FaceDetectorConfig& config);
    bool isInitialized() const;
    void setConfig(const FaceDetectorConfig& config);
    FaceDetectorConfig getConfig() const;
    
    // Main detection methods
    std::vector<FaceDetectionResult> detectFaces(const cv::Mat& image);
    std::vector<FaceDetectionResult> detectFaces(const std::string& imagePath);
    std::vector<FaceDetectionResult> detectFacesMultiScale(const cv::Mat& image);
    

    // Face comparison and recognition
    double compareFaces(const cv::Mat& face1, const cv::Mat& face2);
    double compareFaces(const FaceDetectionResult& face1, const FaceDetectionResult& face2);
    std::vector<std::pair<double, cv::Rect>> findSimilarFaces(const cv::Mat& probeFace, const cv::Mat& image);
    std::vector<std::pair<double, std::string>> findSimilarFacesInFolder(const cv::Mat& probeFace, const std::string& folderPath);

    double compareFacesHistogram(const cv::Mat &face1, const cv::Mat &face2);

    double compareFacesSSIM(const cv::Mat &face1, const cv::Mat &face2);

    double compareFacesFeatures(const cv::Mat &face1, const cv::Mat &face2);

    double compareFacesEmbeddings(const cv::Mat &face1, const cv::Mat &face2);

    cv::Mat calculateLBP(const cv::Mat &src);

    // Batch processing
    std::vector<std::vector<FaceDetectionResult>> detectFacesBatch(
        const std::vector<cv::Mat>& images,
        ProgressCallback callback = nullptr
    );
    std::vector<std::vector<FaceDetectionResult>> detectFacesBatch(
        const std::vector<std::string>& imagePaths,
        ProgressCallback callback = nullptr
    );
    
    // Individual detection methods
    std::vector<cv::Rect> detectFacesDlib(const cv::Mat& image);
    std::vector<cv::Rect> detectFacesHaar(const cv::Mat& image);
    std::vector<cv::Rect> detectFacesLBP(const cv::Mat& image);
    std::vector<cv::Rect> detectFacesDNN(const cv::Mat& image);
    std::vector<cv::Rect> detectFacesYuNet(const cv::Mat& image);
    
    // Landmark detection methods
    std::vector<cv::Point2f> detectLandmarks(const cv::Mat& faceImage, LandmarkMethod method);
    std::vector<cv::Point2f> detectLandmarksDlib(const cv::Mat& faceImage);
    std::vector<cv::Point2f> detectLandmarksLBF(const cv::Mat& faceImage);
    std::vector<cv::Point2f> detectLandmarksYuNet(const cv::Mat& faceImage);
    std::vector<cv::Point2f> detectLandmarksHaar(const cv::Mat& faceImage);
    std::vector<cv::Point2f> generateLandmarks(const cv::Mat& faceImage);

    
    // Face alignment
    cv::Mat alignFace(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    cv::Mat alignFace(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks, 
                      AlignmentMethod method);
    cv::Mat alignFaceSimilarity(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    cv::Mat alignFaceAffine(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    cv::Mat alignFacePerspective(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    cv::Mat alignFaceEyesCenter(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks);
    
    // Face extraction
    cv::Mat extractFace(const cv::Mat& image, const cv::Rect& faceRect, bool addPadding = true);
    std::vector<cv::Mat> extractAllFaces(const cv::Mat& image);
    
    // Post-processing
    std::vector<cv::Rect> filterOverlappingFaces(const std::vector<cv::Rect>& faces, 
                                                   const std::vector<float>& confidences);
    float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
    std::vector<int> nonMaximumSuppression(const std::vector<cv::Rect>& boxes, 
                                            const std::vector<float>& scores, 
                                            float threshold);
    
    // Face quality assessment
    float assessFaceQuality(const cv::Mat& faceImage);
    float calculateSharpness(const cv::Mat& image);
    float calculateBrightness(const cv::Mat& image);
    float calculateContrast(const cv::Mat& image);
    bool isFaceFrontal(const std::vector<cv::Point2f>& landmarks);
    
    // Visualization
    cv::Mat drawFaceDetections(const cv::Mat& image, 
                               const std::vector<FaceDetectionResult>& results,
                               bool showLandmarks = true,
                               bool showConfidence = true);
    cv::Mat drawBoundingBoxes(const cv::Mat& image, const std::vector<cv::Rect>& faces);
    cv::Mat drawLandmarks(const cv::Mat& image, const std::vector<cv::Point2f>& landmarks,
                          bool withNumbers = false);
    
    // Utility methods
    bool loadModels();
    bool loadDlibModel();
    bool loadLBFModel();
    bool loadYuNetModel();
    bool loadDNNModel();
    void releaseModels();
    
    // Statistics
    int getTotalDetections() const;
    double getAverageConfidence() const;
    std::string getDetectionMethodName(DetectionMethod method) const;
    std::string getLandmarkMethodName(LandmarkMethod method) const;
    void resetStatistics();
    
    // Error handling
    std::string getLastError() const;
    bool hasError() const;
    
private:
    // Internal methods
    void preprocessImage(const cv::Mat& input, cv::Mat& output);
    cv::Rect getSafeFaceRect(const cv::Rect &rect, const cv::Size &imageSize);
    cv::Rect expandFaceRect(const cv::Rect &rect, const cv::Size &imageSize, float padding);
    std::vector<cv::Point2f> getEyePositions(const std::vector<cv::Point2f>& landmarks);
    cv::Mat getRotationMatrix(const cv::Point2f& leftEye, const cv::Point2f& rightEye);
    std::vector<cv::Point2f> transformLandmarks(const std::vector<cv::Point2f>& landmarks,
                                                 const cv::Mat& transformMatrix);
    void sortResultsByConfidence(std::vector<FaceDetectionResult>& results);
    float calculateDetectionConfidence(const cv::Rect& face, const cv::Mat& image);
    std::vector<cv::Point2f> generateLandmarksFrom5Points(const std::vector<cv::Point2f>& keyPoints, const cv::Size & faceSize);
    
    // Member variables
    FaceDetectorConfig config_;
    bool initialized_;
    std::string lastError_;
    
    // Statistics
    int totalDetections_;
    double totalConfidence_;
    
    // Dlib models
    std::unique_ptr<dlib::frontal_face_detector> dlibDetector_;
    std::unique_ptr<dlib::shape_predictor> dlibPredictor_;
    
    // OpenCV models
    cv::CascadeClassifier haarCascade_;
    cv::CascadeClassifier lbpCascade_;
    cv::Ptr<cv::face::Facemark> lbfModel_;
    cv::Ptr<cv::FaceDetectorYN> yunetDetector_;
    cv::dnn::Net dnnNet_;
    
    // Thread safety
    mutable std::mutex detectionMutex_;
    
    // Cache
    cv::Mat lastProcessedImage_;
    std::vector<FaceDetectionResult> lastResults_;
};

// Helper functions
namespace FaceDetectionUtils {
    // Convert between coordinate systems
    cv::Rect dlibRectToCV(const dlib::rectangle& rect);
    dlib::rectangle cvRectToDlib(const cv::Rect& rect);
    
    // Landmark conversions
    std::vector<cv::Point2f> dlibPointsToCV(const dlib::full_object_detection& shape);
    
    // Face verification helpers
    bool areSameFace(const FaceDetectionResult& face1, const FaceDetectionResult& face2, 
                     float threshold = 0.6f);
    float calculateFaceSimilarity(const FaceDetectionResult& face1, 
                                  const FaceDetectionResult& face2);
    
    // Image preprocessing
    cv::Mat preprocessForDetection(const cv::Mat& image);
    cv::Mat enhanceContrast(const cv::Mat& image);
    
    // Validation
    bool isValidFaceRect(const cv::Rect& rect, const cv::Size& imageSize);
    bool hasValidLandmarks(const std::vector<cv::Point2f>& landmarks);
    
    // Face pose estimation
    float estimateYaw(const std::vector<cv::Point2f>& landmarks);
    float estimatePitch(const std::vector<cv::Point2f>& landmarks);
    float estimateRoll(const std::vector<cv::Point2f>& landmarks);
}