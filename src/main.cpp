#include "FaceRecognition/FaceDetector.h"
#include "Utils/ImageUtils.h"
#include "../FileSystem/PathUtils.h"
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {
    std::string srcFolder;
    std::string probeImage;
    std::string dstFolder;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-s" && i + 1 < argc) srcFolder = argv[++i];
        else if (arg == "-p" && i + 1 < argc) probeImage = argv[++i];
        else if (arg == "-d" && i + 1 < argc) dstFolder = argv[++i];
    }

    if (srcFolder.empty() || probeImage.empty() || dstFolder.empty()) {
        std::cout << "failed - missing parameters" << std::endl;
        return 1;
    }

 
    if (!PathUtils::fileExists(probeImage)) {
        std::cout << "failed - probe image not found: " << probeImage << std::endl;
        return 1;
    }

    if (!PathUtils::fileExists(srcFolder)) {
        std::cout << "failed - source folder not found: " << srcFolder << std::endl;
        return 1;
    }

    if (!ImageUtils::isValidImage(probeImage)) {
        std::cout << "failed - invalid probe image" << std::endl;
        return 1;
    }

    FaceDetectorConfig config;
    config.detectionMethod = DetectionMethod::DNN_CAFFE;
    config.detectLandmarks = false;
    config.extractFaces = true;
    config.alignFaces = false;
    config.minConfidence = 0.5f;

    FaceDetector detector(config);
    if (!detector.initialize()) {
        std::cout << "failed - detector initialization failed" << std::endl;
        return 1;
    }

    auto probeFaces = detector.detectFaces(probeImage);
    if (probeFaces.empty() || probeFaces[0].faceImage.empty()) {
        std::cout << "failed - no faces found in probe image" << std::endl;
        return 1;
    }

    cv::Mat probeFace = probeFaces[0].faceImage;

try {
    std::filesystem::create_directories(dstFolder);
    std::cout << "Created destination folder: " << dstFolder << std::endl;
} catch (const std::exception& e) {
    std::cout << "failed - cannot create destination folder: " << e.what() << std::endl;
    return 1;
}

    bool copied = false;
    int totalImages = 0;
    int processedImages = 0;
    int similarFacesFound = 0;

  
    for (auto& file : std::filesystem::directory_iterator(srcFolder)) {
        if (file.is_regular_file()) {
            std::string path = file.path().string();
            if (ImageUtils::isValidImage(path)) {
                totalImages++;
            }
        }
    }

    std::cout << "Searching in " << totalImages << " images..." << std::endl;

    for (auto& file : std::filesystem::directory_iterator(srcFolder)) {
        if (!file.is_regular_file()) continue;

        std::string path = file.path().string();
        if (!ImageUtils::isValidImage(path)) continue;

        processedImages++;

    
        if (processedImages % 10 == 0) {
            std::cout << "Processed " << processedImages << "/" << totalImages << " images..." << std::endl;
        }

        auto faces = detector.detectFaces(path);
        for (auto& f : faces) {
            if (f.faceImage.empty()) continue;

            double score = detector.compareFaces(probeFace, f.faceImage);

           
            if (score >= 0.3) {  
                std::cout << "Similarity score: " << score << " in " << PathUtils::getFileName(path) << std::endl;
            }

            if (score >= 5) {      // similarity threshold
                std::string destPath = dstFolder + "/" + file.path().filename().string();
                std::filesystem::copy(path, destPath, std::filesystem::copy_options::overwrite_existing);
                copied = true;
                similarFacesFound++;
                std::cout << "COPIED: " << PathUtils::getFileName(path) << " (score: " << score << ")" << std::endl;
                break;
            }
        }
    }

    std::cout << "Processed " << processedImages << " images, found " << similarFacesFound << " similar faces" << std::endl;
    std::cout << (copied ? "done" : "failed - no similar faces found") << std::endl;
    return 0;
}
