#include "Utils/ImageUtils.h"
#include <iostream>
#include <iomanip>
#include <chrono>

// Helper function to show comparison between original and processed images
void showComparison(const cv::Mat& original, const cv::Mat& processed, const std::string& title) {
    if (original.empty() || processed.empty()) {
        std::cerr << "Cannot show comparison - empty image" << std::endl;
        return;
    }

    cv::Mat orig = original.clone();
    cv::Mat proc = processed.clone();

    // Convert to same number of channels if needed
    if (orig.channels() != proc.channels()) {
        if (orig.channels() == 1 && proc.channels() == 3) {
            cv::cvtColor(orig, orig, cv::COLOR_GRAY2BGR);
        } else if (orig.channels() == 3 && proc.channels() == 1) {
            cv::cvtColor(proc, proc, cv::COLOR_GRAY2BGR);
        }
    }

    // Resize if dimensions don't match
    if (orig.size() != proc.size()) {
        cv::resize(proc, proc, orig.size());
    }

    // Ensure same type
    if (orig.type() != proc.type()) {
        proc.convertTo(proc, orig.type());
    }

    cv::Mat comparison;
    cv::hconcat(orig, proc, comparison);
    cv::imshow(title, comparison);
}

// Helper to measure execution time
template<typename Func>
double measureTime(Func func, const std::string& operation) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << operation << " took " << std::fixed << std::setprecision(2)
              << duration.count() << " ms" << std::endl;
    return duration.count();
}

void printSeparator(const std::string& title = "") {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    if (!title.empty()) {
        std::cout << "  " << title << std::endl;
        std::cout << std::string(70, '=') << std::endl;
    }
}

int main() {
    std::cout << "\nCOMPREHENSIVE IMAGEUTILS TEST SUITE\n" << std::endl;

    const std::string imagePath = "/home/cpluspluser/Desktop/Hussein.jpg";
    const std::string outputDir = "/home/cpluspluser/Desktop/";

    // ========================================================================
    // TEST 1: IMAGE LOADING & VALIDATION
    // ========================================================================
    printSeparator("TEST 1: IMAGE LOADING & VALIDATION");

    std::cout << "Loading image: " << imagePath << std::endl;
    cv::Mat img = ImageUtils::loadImage(imagePath);

    if (img.empty()) {
        std::cout << "FAILED: Could not load image!" << std::endl;
        return -1;
    }
    std::cout << "Image loaded successfully." << std::endl;

    // Validate image
    bool isValid = ImageUtils::isValidImage(imagePath);
    std::cout << "Basic validation: " << (isValid ? "PASSED" : "FAILED") << std::endl;

    bool isValidDeep = ImageUtils::isValidImage(imagePath, true);
    std::cout << "Deep validation: " << (isValidDeep ? "PASSED" : "FAILED") << std::endl;

    bool isBlack = ImageUtils::isImageCompletelyBlack(img);
    std::cout << "Black image check: " << (isBlack ? "FAILED (Black)" : "PASSED (Not Black)") << std::endl;

    // Get image info
    std::string info = ImageUtils::getImageInfo(imagePath);
    std::cout << "\n" << info << std::endl;

    cv::Size imgSize = ImageUtils::getImageSize(img);
    int channels = ImageUtils::getImageChannels(img);
    double quality = ImageUtils::calculateImageQuality(img);

    std::cout << "Dimensions: " << imgSize.width << "x" << imgSize.height << std::endl;
    std::cout << "Channels: " << channels << std::endl;
    std::cout << "Quality Score: " << std::fixed << std::setprecision(2) << quality << "/100" << std::endl;

    bool hasSufficientRes = ImageUtils::hasSufficientResolution(img, 200, 200);
    std::cout << "Resolution check (200x200): " << (hasSufficientRes ? "PASSED" : "FAILED") << std::endl;

    // ========================================================================
    // TEST 2: IMAGE TRANSFORMATION & MANIPULATION
    // ========================================================================
    printSeparator("TEST 2: IMAGE TRANSFORMATION & MANIPULATION");

    std::cout << "\nTesting resize operations..." << std::endl;
    cv::Mat resized;
    measureTime([&]() {
        resized = ImageUtils::resizeImage(img, cv::Size(640, 480));
    }, "Resize to 640x480");
    ImageUtils::saveImage(resized, outputDir + "test_resized_640x480.jpg");
    showComparison(img, resized, "Original vs Resized");

    cv::Mat resizedByWidth;
    measureTime([&]() {
        resizedByWidth = ImageUtils::resizeImageByWidth(img, 800);
    }, "Resize by width (800px)");
    ImageUtils::saveImage(resizedByWidth, outputDir + "test_resized_by_width.jpg");

    cv::Mat resizedByHeight;
    measureTime([&]() {
        resizedByHeight = ImageUtils::resizeImageByHeight(img, 600);
    }, "Resize by height (600px)");
    ImageUtils::saveImage(resizedByHeight, outputDir + "test_resized_by_height.jpg");

    // Crop tests
    std::cout << "\nTesting crop operations..." << std::endl;
    cv::Rect centerCrop(img.cols / 4, img.rows / 4, img.cols / 2, img.rows / 2);
    cv::Mat cropped = ImageUtils::cropImage(img, centerCrop);
    ImageUtils::saveImage(cropped, outputDir + "test_cropped_center.jpg");
    std::cout << "Center crop saved." << std::endl;

    cv::Rect outOfBoundsCrop(-50, -50, img.cols + 100, img.rows + 100);
    cv::Mat croppedSafe = ImageUtils::cropImageSafe(img, outOfBoundsCrop, cv::Scalar(128, 128, 128));
    ImageUtils::saveImage(croppedSafe, outputDir + "test_cropped_safe.jpg");
    std::cout << "Safe crop with padding saved." << std::endl;

    // Rotation tests
    std::cout << "\nTesting rotation operations..." << std::endl;
    cv::Mat rotated45, rotated90, rotated180;
    measureTime([&]() {
        rotated45 = ImageUtils::rotateImage(img, 45);
    }, "Rotate 45°");
    measureTime([&]() {
        rotated90 = ImageUtils::rotateImage(img, 90);
    }, "Rotate 90°");
    measureTime([&]() {
        rotated180 = ImageUtils::rotateImage(img, 180);
    }, "Rotate 180°");

    ImageUtils::saveImage(rotated45, outputDir + "test_rotated_45.jpg");
    ImageUtils::saveImage(rotated90, outputDir + "test_rotated_90.jpg");
    ImageUtils::saveImage(rotated180, outputDir + "test_rotated_180.jpg");

    // Flip tests
    std::cout << "\nTesting flip operations..." << std::endl;
    cv::Mat flippedH = ImageUtils::flipHorizontal(img);
    cv::Mat flippedV = ImageUtils::flipVertically(img);
    ImageUtils::saveImage(flippedH, outputDir + "test_flipped_horizontal.jpg");
    ImageUtils::saveImage(flippedV, outputDir + "test_flipped_vertical.jpg");
    showComparison(img, flippedH, "Original vs Horizontal Flip");
    std::cout << "Horizontal flip saved." << std::endl;
    std::cout << "Vertical flip saved." << std::endl;

    // ========================================================================
    // TEST 3: COLOR & IMAGE PROCESSING
    // ========================================================================
    printSeparator("TEST 3: COLOR & IMAGE PROCESSING");

    std::cout << "\nConverting to grayscale..." << std::endl;
    cv::Mat gray;
    measureTime([&]() {
        gray = ImageUtils::convertToGray(img);
    }, "Convert to grayscale");
    ImageUtils::saveImage(gray, outputDir + "test_grayscale.jpg");
    showComparison(img, gray, "Original vs Grayscale");

    std::cout << "\nApplying histogram equalization..." << std::endl;
    cv::Mat equalized;
    measureTime([&]() {
        equalized = ImageUtils::equalizeHistogram(img);
    }, "Histogram equalization");
    ImageUtils::saveImage(equalized, outputDir + "test_equalized.jpg");
    showComparison(img, equalized, "Original vs Equalized");

    std::cout << "\nNormalizing image..." << std::endl;
    cv::Mat normalized;
    measureTime([&]() {
        normalized = ImageUtils::normalizeImage(gray);
    }, "Normalize image");
    ImageUtils::saveImage(normalized, outputDir + "test_normalized.jpg");

    std::cout << "\nAdjusting brightness and contrast..." << std::endl;
    cv::Mat brighterContrast = ImageUtils::adjustBrightnessContrast(img, 1.3, 30);
    cv::Mat darkerLowContrast = ImageUtils::adjustBrightnessContrast(img, 0.7, -20);
    ImageUtils::saveImage(brighterContrast, outputDir + "test_bright_contrast.jpg");
    ImageUtils::saveImage(darkerLowContrast, outputDir + "test_dark_lowcontrast.jpg");
    showComparison(img, brighterContrast, "Original vs Brighter+Contrast");
    std::cout << "Brightness and contrast adjustments saved." << std::endl;

    // ========================================================================
    // TEST 4: FACE DETECTION & PROCESSING
    // ========================================================================
    printSeparator("TEST 4: FACE DETECTION & PROCESSING");

    std::cout << "\nDetecting faces..." << std::endl;
    cv::Mat grayFace = ImageUtils::convertToGray(img);
    std::vector<cv::Rect> faces;
    measureTime([&]() {
        faces = ImageUtils::detectFacesEnhanced(grayFace);
    }, "Face detection");

    std::cout << "Detected " << faces.size() << " face(s)." << std::endl;

    cv::Mat imgWithFaces = img.clone();
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(imgWithFaces, faces[i], cv::Scalar(0, 255, 0), 3);
        cv::putText(imgWithFaces, "Face " + std::to_string(i + 1),
                    cv::Point(faces[i].x, faces[i].y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    }
    ImageUtils::saveImage(imgWithFaces, outputDir + "test_faces_detected.jpg");
    cv::imshow("Detected Faces", imgWithFaces);

    // ========================================================================
    // TEST 5: FACIAL LANDMARKS DETECTION
    // ========================================================================
    printSeparator("TEST 5: FACIAL LANDMARKS DETECTION");

    cv::Mat faceForLandmarks;
    if (!faces.empty()) {
        faceForLandmarks = ImageUtils::extractFace(img, faces[0], true, 0.2);
        ImageUtils::saveImage(faceForLandmarks, outputDir + "test_extracted_face.jpg");
        std::cout << "Face extracted with padding." << std::endl;
    } else {
        faceForLandmarks = img.clone();
        std::cout << "No faces detected, using full image." << std::endl;
    }

    std::cout << "\nTesting landmark detection methods..." << std::endl;

    std::vector<cv::Point2f> landmarksAuto, landmarksDlib, landmarksLBF, landmarksYuNet, landmarksHAAR;
    std::vector<cv::Point2f> landmarksEnhanced, landmarksDefault;

    measureTime([&]() { landmarksAuto = ImageUtils::detectFacialLandmarks(faceForLandmarks); }, "Auto landmark detection");
    std::cout << "Detected " << landmarksAuto.size() << " landmarks (Auto)." << std::endl;

    measureTime([&]() { landmarksDlib = ImageUtils::detectLandmarksDlib(faceForLandmarks); }, "Dlib landmark detection");
    std::cout << "Detected " << landmarksDlib.size() << " landmarks (Dlib)." << std::endl;

    measureTime([&]() { landmarksLBF = ImageUtils::detectLandmarksLBF(faceForLandmarks); }, "LBF landmark detection");
    std::cout << "Detected " << landmarksLBF.size() << " landmarks (LBF)." << std::endl;

    measureTime([&]() { landmarksYuNet = ImageUtils::detectFacialLandmarksYuNet(faceForLandmarks); }, "YuNet landmark detection");
    std::cout << "Detected " << landmarksYuNet.size() << " landmarks (YuNet)." << std::endl;

    measureTime([&]() { landmarksHAAR = ImageUtils::detectLandmarksHAAR(faceForLandmarks); }, "HAAR landmark detection");
    std::cout << "Detected " << landmarksHAAR.size() << " landmarks (HAAR)." << std::endl;

    measureTime([&]() { landmarksEnhanced = ImageUtils::getEnhancedFaceLandmarks(faceForLandmarks); }, "Enhanced landmark detection");
    std::cout << "Generated " << landmarksEnhanced.size() << " landmarks (Enhanced)." << std::endl;

    measureTime([&]() { landmarksDefault = ImageUtils::getDefaultFaceLandmarks(faceForLandmarks); }, "Default landmark generation");
    std::cout << "Generated " << landmarksDefault.size() << " landmarks (Default)." << std::endl;

    // Draw landmarks with detailed annotations
    std::cout << "\nDrawing landmarks..." << std::endl;
    cv::Mat imgWithLandmarks = faceForLandmarks.clone();
    std::vector<cv::Point2f>& landmarks = landmarksAuto.empty() ? landmarksEnhanced : landmarksAuto;

    for (size_t i = 0; i < landmarks.size(); i++) {
        cv::Scalar color;
        if (i < 17) color = cv::Scalar(255, 0, 0);        // Jaw line
        else if (i < 27) color = cv::Scalar(0, 255, 0);   // Eyebrows
        else if (i < 36) color = cv::Scalar(0, 0, 255);   // Nose
        else if (i < 48) color = cv::Scalar(255, 255, 0); // Eyes
        else color = cv::Scalar(255, 0, 255);             // Mouth

        cv::circle(imgWithLandmarks, landmarks[i], 3, color, -1);
        cv::circle(imgWithLandmarks, landmarks[i], 5, cv::Scalar(255, 255, 255), 1);

        if (i < 20) {
            cv::putText(imgWithLandmarks, std::to_string(i + 1),
                        landmarks[i] + cv::Point2f(6, -6),
                        cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
        }
    }

    ImageUtils::saveImageWithQuality(imgWithLandmarks, outputDir + "test_landmarks_detailed.jpg", 100);
    cv::imshow("Facial Landmarks (Detailed)", imgWithLandmarks);

    // Draw connections between landmarks
    cv::Mat imgWithConnections = faceForLandmarks.clone();
    for (size_t i = 0; i < landmarks.size(); i++) {
        cv::circle(imgWithConnections, landmarks[i], 2, cv::Scalar(0, 255, 0), -1);
    }

    for (size_t i = 0; i < 16 && i < landmarks.size() - 1; i++)
        cv::line(imgWithConnections, landmarks[i], landmarks[i + 1], cv::Scalar(255, 0, 0), 1);

    for (size_t i = 17; i < 21 && i < landmarks.size() - 1; i++)
        cv::line(imgWithConnections, landmarks[i], landmarks[i + 1], cv::Scalar(0, 255, 0), 1);
    for (size_t i = 22; i < 26 && i < landmarks.size() - 1; i++)
        cv::line(imgWithConnections, landmarks[i], landmarks[i + 1], cv::Scalar(0, 255, 0), 1);

    ImageUtils::saveImage(imgWithConnections, outputDir + "test_landmarks_connected.jpg");
    cv::imshow("Facial Landmarks (Connected)", imgWithConnections);

    // ========================================================================
    // TEST 6: SAVE WITH DIFFERENT QUALITIES
    // ========================================================================
    printSeparator("TEST 6: SAVE WITH DIFFERENT QUALITIES");

    std::cout << "\nSaving images with different quality settings..." << std::endl;
    ImageUtils::saveImageWithQuality(img, outputDir + "test_quality_100.jpg", 100);
    ImageUtils::saveImageWithQuality(img, outputDir + "test_quality_75.jpg", 75);
    ImageUtils::saveImageWithQuality(img, outputDir + "test_quality_50.jpg", 50);
    ImageUtils::saveImageWithQuality(img, outputDir + "test_quality_25.jpg", 25);
    std::cout << "Saved 4 versions with different qualities." << std::endl;

    // ========================================================================
    // TEST 7: INTERPOLATION METHODS
    // ========================================================================
    printSeparator("TEST 7: INTERPOLATION METHODS");

    std::cout << "\nTesting interpolation selection..." << std::endl;
    cv::Size originalSize = img.size();
    cv::Size smallSize(320, 240);
    cv::Size largeSize(1920, 1080);

    int interpSmall = ImageUtils::getOptimalInterpolation(originalSize, smallSize);
    int interpLarge = ImageUtils::getOptimalInterpolation(originalSize, largeSize);

    std::cout << "Downsampling (" << originalSize << " → " << smallSize << "): "
              << ImageUtils::getInterpolationName(interpSmall) << std::endl;
    std::cout << "Upsampling (" << originalSize << " → " << largeSize << "): "
              << ImageUtils::getInterpolationName(interpLarge) << std::endl;

    // ========================================================================
    // FINAL SUMMARY
    // ========================================================================
    printSeparator("TEST SUMMARY");

    std::cout << "\nAll test outputs saved to: " << outputDir << std::endl;
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "   - test_resized_*.jpg (3 files)" << std::endl;
    std::cout << "   - test_cropped_*.jpg (2 files)" << std::endl;
    std::cout << "   - test_rotated_*.jpg (3 files)" << std::endl;
    std::cout << "   - test_flipped_*.jpg (2 files)" << std::endl;
    std::cout << "   - test_grayscale.jpg" << std::endl;
    std::cout << "   - test_equalized.jpg" << std::endl;
    std::cout << "   - test_normalized.jpg" << std::endl;
    std::cout << "   - test_bright_contrast.jpg" << std::endl;
    std::cout << "   - test_dark_lowcontrast.jpg" << std::endl;
    std::cout << "   - test_faces_detected.jpg" << std::endl;
    std::cout << "   - test_extracted_face.jpg" << std::endl;
    std::cout << "   - test_landmarks_detailed.jpg" << std::endl;
    std::cout << "   - test_landmarks_connected.jpg" << std::endl;
    std::cout << "   - test_quality_*.jpg (4 files)" << std::endl;

    std::cout << "\nTotal: approximately 25+ test output files" << std::endl;
    std::cout << "\nAll tests completed successfully." << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
