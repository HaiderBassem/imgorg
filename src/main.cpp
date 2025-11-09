#include "FaceRecognition/FaceDetector.h"
#include "Utils/ImageUtils.h"
#include <iostream>

int main() {
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║     Face Detection Test - Fixed       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝" << std::endl;

    // المسار الصحيح للصورة
    std::string imagePath = "/home/cpluspluser/Desktop/Alsaeed.jpg";

    // 
    std::cout << "\n[1] Checking image..." << std::endl;
    if (!ImageUtils::isValidImage(imagePath)) {
        std::cerr << "[ERROR] Image not found or invalid: " << imagePath << std::endl;
        return 1;
    }
    std::cout << "[✓] Image found and valid" << std::endl;
    std::cout << ImageUtils::getImageInfo(imagePath) << std::endl;

    // 2. إعداد الإعدادات (استخدام DNN فقط لأنه الوحيد الشغال)
    std::cout << "\n[2] Configuring detector..." << std::endl;
    FaceDetectorConfig config;
    config.detectionMethod = DetectionMethod::DNN_CAFFE;  // استخدم DNN لأنه شغال
    config.detectLandmarks = false;  // عطل landmarks لأن models مو موجودة
    config.minConfidence = 0.5f;
    config.extractFaces = true;
    config.alignFaces = false;  // عطل alignment لأن landmarks معطلة

    std::cout << "[✓] Using DNN_CAFFE detection (most reliable without models)" << std::endl;

    // 3. إنشاء وتهيئة الكاشف
    std::cout << "\n[3] Initializing detector..." << std::endl;
    FaceDetector detector(config);

    if (!detector.initialize()) {
        std::cerr << "[ERROR] Failed to initialize detector" << std::endl;
        if (detector.hasError()) {
            std::cerr << "Error: " << detector.getLastError() << std::endl;
        }
        return 1;
    }
    std::cout << "[✓] Detector initialized successfully" << std::endl;

    // 4. كشف الوجوه
    std::cout << "\n[4] Detecting faces..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    auto results = detector.detectFaces(imagePath);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "[✓] Detection completed in " << duration.count() << " ms" << std::endl;
    std::cout << "[✓] Found " << results.size() << " face(s)" << std::endl;

    if (results.empty()) {
        std::cout << "\n[WARNING] No faces detected!" << std::endl;
        std::cout << "Try:" << std::endl;
        std::cout << "  - Lowering minConfidence" << std::endl;
        std::cout << "  - Using a different image" << std::endl;
        std::cout << "  - Checking image quality" << std::endl;
        return 0;
    }

    // 5. طباعة تفاصيل الوجوه المكتشفة
    std::cout << "\n[5] Face Details:" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& face = results[i];
        std::cout << "   Face #" << (i + 1) << ":" << std::endl;
        std::cout << "     - Position: (" << face.boundingBox.x << ", "
                  << face.boundingBox.y << ")" << std::endl;
        std::cout << "     - Size: " << face.boundingBox.width << "x"
                  << face.boundingBox.height << " pixels" << std::endl;
        std::cout << "     - Confidence: " << (face.confidence * 100.0f) << "%" << std::endl;
    }

    // 6. تحميل الصورة الأصلية (نفس المسار!)
    std::cout << "\n[6] Loading image for visualization..." << std::endl;
    cv::Mat image = ImageUtils::loadImage(imagePath);

    if (image.empty()) {
        std::cerr << "[ERROR] Failed to load image for visualization" << std::endl;
        return 1;
    }
    std::cout << "[✓] Image loaded: " << image.cols << "x" << image.rows << std::endl;

    // 7. رسم النتائج
    std::cout << "\n[7] Drawing detections..." << std::endl;
    cv::Mat output = detector.drawFaceDetections(image, results, false, true);

    if (output.empty()) {
        std::cerr << "[ERROR] Failed to draw detections" << std::endl;
        return 1;
    }
    std::cout << "[✓] Detections drawn successfully" << std::endl;

    // 8. حفظ النتيجة
    std::cout << "\n[8] Saving results..." << std::endl;
    std::string outputPath = "/home/cpluspluser/Desktop/Hussein_detected.jpg";

    if (!ImageUtils::saveImage(output, outputPath)) {
        std::cerr << "[ERROR] Failed to save output image" << std::endl;
        return 1;
    }
    std::cout << "[✓] Results saved to: " << outputPath << std::endl;

    // 9. حفظ الوجوه المستخرجة
    std::cout << "\n[9] Saving extracted faces..." << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].faceImage.empty()) {
            std::string facePath = "/home/cpluspluser/Desktop/Hussein_face_" +
                                   std::to_string(i + 1) + ".jpg";

            if (ImageUtils::saveImage(results[i].faceImage, facePath)) {
                std::cout << "   [✓] Face #" << (i + 1) << " saved to: " << facePath << std::endl;
            }
        }
    }

    // 10. إحصائيات
    std::cout << "\n╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║           Statistics                   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝" << std::endl;
    std::cout << "Total detections: " << detector.getTotalDetections() << std::endl;
    std::cout << "Average confidence: " << (detector.getAverageConfidence() * 100) << "%" << std::endl;
    std::cout << "Detection time: " << duration.count() << " ms" << std::endl;

    std::cout << "\n[✓✓✓] All operations completed successfully! [✓✓✓]" << std::endl;

    return 0;
}
