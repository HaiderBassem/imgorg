#include "src/FaceRecognition/FaceDetector.h"
#include "src/Utils/ImageUtils.h"
#include <iostream>

int main() {
    std::cout << "==================================" << std::endl;
    std::cout << "  Face Detection System v2.0" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // ============================================================================
    // EXAMPLE 1: Basic Image Operations with ImageUtils
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 1] Testing ImageUtils..." << std::endl;
    
    std::string imagePath = "test_image.jpg";
    
    // Load image
    cv::Mat image = ImageUtils::loadImage(imagePath);
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }
    
    // Get image info
    std::cout << ImageUtils::getImageInfo(imagePath) << std::endl;
    
    // Image transformations
    cv::Mat resized = ImageUtils::resizeImage(image, cv::Size(800, 600));
    cv::Mat gray = ImageUtils::convertToGray(image);
    cv::Mat equalized = ImageUtils::equalizeHistogram(image);
    cv::Mat rotated = ImageUtils::rotateImage(image, 45.0);
    cv::Mat flipped = ImageUtils::flipHorizontal(image);
    
    // Save processed images
    ImageUtils::saveImage(resized, "output_resized.jpg");
    ImageUtils::saveImage(gray, "output_gray.jpg");
    ImageUtils::saveImage(equalized, "output_equalized.jpg");
    
    std::cout << "✓ Image operations completed!" << std::endl;
    
    // ============================================================================
    // EXAMPLE 2: Basic Face Detection
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 2] Testing Basic Face Detection..." << std::endl;
    
    // Create detector with default configuration
    FaceDetector detector;
    
    // Initialize
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize detector!" << std::endl;
        return -1;
    }
    
    // Detect faces
    auto results = detector.detectFaces(image);
    
    std::cout << "Found " << results.size() << " face(s)" << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& result = results[i];
        std::cout << "\nFace #" << (i + 1) << ":" << std::endl;
        std::cout << "  Position: (" << result.boundingBox.x << ", " << result.boundingBox.y << ")" << std::endl;
        std::cout << "  Size: " << result.boundingBox.width << "x" << result.boundingBox.height << std::endl;
        std::cout << "  Confidence: " << (result.confidence * 100) << "%" << std::endl;
        std::cout << "  Landmarks: " << result.landmarks.size() << " points" << std::endl;
        
        // Save individual face
        if (!result.faceImage.empty()) {
            std::string faceFile = "face_" + std::to_string(i + 1) + ".jpg";
            ImageUtils::saveImage(result.faceImage, faceFile);
        }
        
        // Save aligned face
        if (!result.alignedFace.empty()) {
            std::string alignedFile = "aligned_face_" + std::to_string(i + 1) + ".jpg";
            ImageUtils::saveImage(result.alignedFace, alignedFile);
        }
    }
    
    // Draw detections on image
    cv::Mat visualized = detector.drawFaceDetections(image, results, true, true);
    ImageUtils::saveImage(visualized, "output_detected.jpg");
    
    // ============================================================================
    // EXAMPLE 3: Custom Configuration
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 3] Testing Custom Configuration..." << std::endl;
    
    FaceDetectorConfig config;
    config.detectionMethod = DetectionMethod::YUNET;
    config.landmarkMethod = LandmarkMethod::DLIB_68;
    config.minConfidence = 0.7f;
    config.alignFaces = true;
    config.alignmentMethod = AlignmentMethod::SIMILARITY;
    config.alignedFaceSize = cv::Size(256, 256);
    config.extractFaces = true;
    config.detectLandmarks = true;
    
    FaceDetector customDetector(config);
    
    if (customDetector.initialize()) {
        auto customResults = customDetector.detectFaces(image);
        std::cout << "Custom detector found " << customResults.size() << " face(s)" << std::endl;
    }
    
    // ============================================================================
    // EXAMPLE 4: Multi-Scale Detection
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 4] Testing Multi-Scale Detection..." << std::endl;
    
    config.enableMultiScale = true;
    FaceDetector multiScaleDetector(config);
    
    if (multiScaleDetector.initialize()) {
        auto multiScaleResults = multiScaleDetector.detectFacesMultiScale(image);
        std::cout << "Multi-scale detector found " << multiScaleResults.size() << " face(s)" << std::endl;
    }
    
    // ============================================================================
    // EXAMPLE 5: Batch Processing
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 5] Testing Batch Processing..." << std::endl;
    
    std::vector<std::string> imagePaths = {
        "image1.jpg",
        "image2.jpg",
        "image3.jpg"
    };
    
    auto batchResults = detector.detectFacesBatch(imagePaths, 
        [](int current, int total, const std::string& msg) {
            std::cout << "Progress: " << current << "/" << total << " - " << msg << std::endl;
        });
    
    std::cout << "Batch processing completed for " << batchResults.size() << " images" << std::endl;
    
    int totalFaces = 0;
    for (const auto& results : batchResults) {
        totalFaces += results.size();
    }
    std::cout << "Total faces detected: " << totalFaces << std::endl;
    
    // ============================================================================
    // EXAMPLE 6: Quality Assessment
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 6] Testing Face Quality Assessment..." << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].faceImage.empty()) {
            float quality = detector.assessFaceQuality(results[i].faceImage);
            float sharpness = detector.calculateSharpness(results[i].faceImage);
            float brightness = detector.calculateBrightness(results[i].faceImage);
            float contrast = detector.calculateContrast(results[i].faceImage);
            bool frontal = detector.isFaceFrontal(results[i].landmarks);
            
            std::cout << "\nFace #" << (i + 1) << " Quality:" << std::endl;
            std::cout << "  Overall Quality: " << (quality * 100) << "/100" << std::endl;
            std::cout << "  Sharpness: " << sharpness << std::endl;
            std::cout << "  Brightness: " << (brightness * 100) << "%" << std::endl;
            std::cout << "  Contrast: " << contrast << std::endl;
            std::cout << "  Is Frontal: " << (frontal ? "Yes" : "No") << std::endl;
        }
    }
    
    // ============================================================================
    // EXAMPLE 7: Pose Estimation
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 7] Testing Pose Estimation..." << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].landmarks.empty()) {
            float yaw = FaceDetectionUtils::estimateYaw(results[i].landmarks);
            float pitch = FaceDetectionUtils::estimatePitch(results[i].landmarks);
            float roll = FaceDetectionUtils::estimateRoll(results[i].landmarks);
            
            std::cout << "\nFace #" << (i + 1) << " Pose:" << std::endl;
            std::cout << "  Yaw: " << yaw << "°" << std::endl;
            std::cout << "  Pitch: " << pitch << "°" << std::endl;
            std::cout << "  Roll: " << roll << "°" << std::endl;
        }
    }
    
    // ============================================================================
    // EXAMPLE 8: Different Detection Methods
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 8] Testing Different Detection Methods..." << std::endl;
    
    std::vector<DetectionMethod> methods = {
        DetectionMethod::DLIB,
        DetectionMethod::HAAR_CASCADE,
        DetectionMethod::DNN_CAFFE,
        DetectionMethod::YUNET,
        DetectionMethod::AUTO
    };
    
    for (auto method : methods) {
        FaceDetectorConfig testConfig;
        testConfig.detectionMethod = method;
        testConfig.detectLandmarks = false; // Faster
        
        FaceDetector testDetector(testConfig);
        
        if (testDetector.initialize()) {
            auto testResults = testDetector.detectFaces(image);
            std::cout << testDetector.getDetectionMethodName(method) 
                      << " found " << testResults.size() << " face(s)" << std::endl;
        }
    }
    
    // ============================================================================
    // EXAMPLE 9: Landmark Visualization
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 9] Testing Landmark Visualization..." << std::endl;
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].landmarks.empty()) {
            cv::Mat landmarksVis = detector.drawLandmarks(
                image, 
                results[i].landmarks, 
                true  // Show numbers
            );
            
            std::string landmarksFile = "landmarks_face_" + std::to_string(i + 1) + ".jpg";
            ImageUtils::saveImage(landmarksVis, landmarksFile);
        }
    }
    
    // ============================================================================
    // EXAMPLE 10: Statistics
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 10] Detector Statistics..." << std::endl;
    
    std::cout << "Total detections: " << detector.getTotalDetections() << std::endl;
    std::cout << "Average confidence: " << (detector.getAverageConfidence() * 100) << "%" << std::endl;
    
    // Reset statistics
    detector.resetStatistics();
    
    // ============================================================================
    // EXAMPLE 11: Face Alignment Methods
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 11] Testing Different Alignment Methods..." << std::endl;
    
    if (!results.empty() && !results[0].landmarks.empty()) {
        std::vector<AlignmentMethod> alignMethods = {
            AlignmentMethod::SIMILARITY,
            AlignmentMethod::AFFINE,
            AlignmentMethod::PERSPECTIVE,
            AlignmentMethod::EYES_CENTER
        };
        
        for (auto method : alignMethods) {
            cv::Mat aligned = detector.alignFace(image, results[0].landmarks, method);
            
            if (!aligned.empty()) {
                std::string filename = "aligned_";
                switch (method) {
                    case AlignmentMethod::SIMILARITY:
                        filename += "similarity.jpg";
                        break;
                    case AlignmentMethod::AFFINE:
                        filename += "affine.jpg";
                        break;
                    case AlignmentMethod::PERSPECTIVE:
                        filename += "perspective.jpg";
                        break;
                    case AlignmentMethod::EYES_CENTER:
                        filename += "eyes_center.jpg";
                        break;
                }
                
                ImageUtils::saveImage(aligned, filename);
                std::cout << "✓ Saved " << filename << std::endl;
            }
        }
    }
    
    // ============================================================================
    // EXAMPLE 12: Extract All Faces
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 12] Extracting All Faces..." << std::endl;
    
    std::vector<cv::Mat> allFaces = detector.extractAllFaces(image);
    
    std::cout << "Extracted " << allFaces.size() << " face(s)" << std::endl;
    
    for (size_t i = 0; i < allFaces.size(); ++i) {
        std::string faceFile = "extracted_face_" + std::to_string(i + 1) + ".jpg";
        ImageUtils::saveImage(allFaces[i], faceFile);
    }
    
    // ============================================================================
    // EXAMPLE 13: Error Handling
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 13] Testing Error Handling..." << std::endl;
    
    // Try to detect on invalid image
    cv::Mat emptyImage;
    auto errorResults = detector.detectFaces(emptyImage);
    
    if (detector.hasError()) {
        std::cout << "Error detected: " << detector.getLastError() << std::endl;
    }
    
    // Try to load invalid image
    cv::Mat invalidImage = ImageUtils::loadImage("nonexistent.jpg");
    if (invalidImage.empty()) {
        std::cout << "Successfully handled invalid image path" << std::endl;
    }
    
    // ============================================================================
    // EXAMPLE 14: Image Validation
    // ============================================================================
    
    std::cout << "\n[EXAMPLE 14] Testing Image Validation..." << std::endl;
    
    bool isValid = ImageUtils::isValidImage(imagePath, true);
    std::cout << "Image validation result: " << (isValid ? "VALID" : "INVALID") << std::endl;
    
    double quality = ImageUtils::calculateImageQuality(image);
    std::cout << "Image quality score: " << quality << "/100" << std::endl;
    
    bool hasSufficientRes = ImageUtils::hasSufficientResolution(image, 640, 480);
    std::cout << "Has sufficient resolution (640x480): " << (hasSufficientRes ? "Yes" : "No") << std::endl;
    
    // ============================================================================
    // FINAL SUMMARY
    // ============================================================================
    
    std::cout << "\n==================================" << std::endl;
    std::cout << "  All Examples Completed!" << std::endl;
    std::cout << "==================================" << std::endl;
    
    std::cout << "\nFinal Statistics:" << std::endl;
    std::cout << "- Total faces detected: " << results.size() << std::endl;
    std::cout << "- Average confidence: " << (detector.getAverageConfidence() * 100) << "%" << std::endl;
    std::cout << "- Detection method: " << detector.getDetectionMethodName(config.detectionMethod) << std::endl;
    std::cout << "- Landmark method: " << detector.getLandmarkMethodName(config.landmarkMethod) << std::endl;
    
    return 0;
}