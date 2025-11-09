/*
 *
 * Haider Bassem Nassif
 * Face Organizer v1.0 - 2025-11-9
 * 
 * 
 */

#include "FaceRecognition/FaceDetector.h"
#include "Utils/ImageUtils.h"
#include "FileSystem/PathUtils.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <vector>
#include <string>


// CONFIGURATION

struct Config {
    std::string probeImagePath;
    std::string sourceFolderPath;
    std::string destFolderPath;
    float threshold = 0.65f;
    bool verbose = false;
    int numThreads = -1;
};


// STATISTICS

struct Statistics 
{
    int totalImages = 0;
    int processedImages = 0;
    int matchedImages = 0;
    int totalFaces = 0;
    int skippedImages = 0;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    
    double totalSeconds() const 
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        return duration.count() / 1000.0;
    }
    
    void print() const 
    {
        std::cout << " Results:\n";
        std::cout << "   Total images:    " << totalImages << "\n";
        std::cout << "   Processed:       " << processedImages << "\n";
        std::cout << "   Matched:         " << matchedImages << " (" 
                  << std::fixed << std::setprecision(1)
                  << (totalImages > 0 ? (matchedImages * 100.0 / totalImages) : 0) << "%)\n";
        std::cout << "   Skipped:         " << skippedImages << "\n";
        std::cout << "   Faces detected:  " << totalFaces << "\n\n";
        std::cout << "   Performance:\n";
        std::cout << "   Total time:      " << std::fixed << std::setprecision(2) << totalSeconds() << " seconds\n";
        std::cout << "   Speed:           " << std::fixed << std::setprecision(2) 
                  << (processedImages > 0 ? processedImages / totalSeconds() : 0) << " images/sec\n";
        std::cout << "\n" << std::string(56, '=') << "\n";
    }
};


// ENHANCED FACE COMPARISON (Inline Implementation)

class EnhancedComparator 
{
public:
    // Main comparison using multiple methods
    static double compareFacesAdvanced(const cv::Mat& face1, const cv::Mat& face2) 
    {
        if (face1.empty() || face2.empty()) return 0.0;
        
        cv::Mat f1, f2;
        cv::resize(face1, f1, cv::Size(160, 160));
        cv::resize(face2, f2, cv::Size(160, 160));
        
        // Method 1: Multi-channel histogram (40% weight)
        double histScore = compareHistogram(f1, f2);
        
        // Method 2: LBP texture (30% weight)
        double lbpScore = compareLBP(f1, f2);
        
        // Method 3: SSIM structure (20% weight)
        double ssimScore = compareSSIM(f1, f2);
        
        // Method 4: Color moments (10% weight)
        double momentScore = compareColorMoments(f1, f2);
        
        // Weighted combination
        double finalScore = histScore * 0.40 + 
                          lbpScore * 0.30 + 
                          ssimScore * 0.20 + 
                          momentScore * 0.10;
        
        return std::max(0.0, std::min(1.0, finalScore));
    }
    
private:
    // Histogram comparison in Lab color space
    static double compareHistogram(const cv::Mat& f1, const cv::Mat& f2) 
    {
        cv::Mat lab1, lab2;
        cv::cvtColor(f1, lab1, cv::COLOR_BGR2Lab);
        cv::cvtColor(f2, lab2, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> channels1, channels2;
        cv::split(lab1, channels1);
        cv::split(lab2, channels2);
        
        double totalScore = 0.0;
        int histSize = 32;
        float range[] = {0, 256};
        const float* histRange = {range};
        
        for (size_t i = 0; i < 3; i++) 
        {
            cv::Mat hist1, hist2;
            cv::calcHist(&channels1[i], 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
            cv::calcHist(&channels2[i], 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);
            
            cv::normalize(hist1, hist1, 1.0, 0.0, cv::NORM_L1);
            cv::normalize(hist2, hist2, 1.0, 0.0, cv::NORM_L1);
            
            double correl = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
            totalScore += (correl + 1.0) / 2.0;
        }
        
        return totalScore / 3.0;
    }
    
    // LBP texture comparison
    static double compareLBP(const cv::Mat& f1, const cv::Mat& f2) 
    {
        cv::Mat gray1, gray2;
        cv::cvtColor(f1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(f2, gray2, cv::COLOR_BGR2GRAY);
        
        cv::Mat lbp1 = calculateLBP(gray1);
        cv::Mat lbp2 = calculateLBP(gray2);
        
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        
        cv::Mat hist1, hist2;
        cv::calcHist(&lbp1, 1, 0, cv::Mat(), hist1, 1, &histSize, &histRange);
        cv::calcHist(&lbp2, 1, 0, cv::Mat(), hist2, 1, &histSize, &histRange);
        
        cv::normalize(hist1, hist1, 1.0, 0.0, cv::NORM_L1);
        cv::normalize(hist2, hist2, 1.0, 0.0, cv::NORM_L1);
        
        double correl = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
        return (correl + 1.0) / 2.0;
    }
    
    // SSIM comparison
    static double compareSSIM(const cv::Mat& f1, const cv::Mat& f2) 
    {
        cv::Mat gray1, gray2;
        cv::cvtColor(f1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(f2, gray2, cv::COLOR_BGR2GRAY);
        
        cv::Mat I1, I2;
        gray1.convertTo(I1, CV_32F);
        gray2.convertTo(I2, CV_32F);
        
        const double C1 = 6.5025, C2 = 58.5225;
        
        cv::Mat mu1, mu2;
        cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
        cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
        
        cv::Mat mu1_2 = mu1.mul(mu1);
        cv::Mat mu2_2 = mu2.mul(mu2);
        cv::Mat mu1_mu2 = mu1.mul(mu2);
        
        cv::Mat sigma1_2, sigma2_2, sigma12;
        cv::GaussianBlur(I1.mul(I1), sigma1_2, cv::Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
        
        cv::GaussianBlur(I2.mul(I2), sigma2_2, cv::Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
        
        cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        
        cv::Mat t1 = 2 * mu1_mu2 + C1;
        cv::Mat t2 = 2 * sigma12 + C2;
        cv::Mat numerator = t1.mul(t2);
        
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        cv::Mat denominator = t1.mul(t2);
        
        cv::Mat ssim_map;
        cv::divide(numerator, denominator, ssim_map);
        
        return cv::mean(ssim_map)[0];
    }
    
    // Color moments comparison
    static double compareColorMoments(const cv::Mat& f1, const cv::Mat& f2) 
    {
        cv::Mat lab1, lab2;
        cv::cvtColor(f1, lab1, cv::COLOR_BGR2Lab);
        cv::cvtColor(f2, lab2, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> ch1, ch2;
        cv::split(lab1, ch1);
        cv::split(lab2, ch2);
        
        double totalDiff = 0.0;
        
        for (size_t i = 0; i < 3; i++) {
            cv::Scalar mean1, stddev1, mean2, stddev2;
            cv::meanStdDev(ch1[i], mean1, stddev1);
            cv::meanStdDev(ch2[i], mean2, stddev2);
            
            double meanDiff = std::abs(mean1[0] - mean2[0]) / 255.0;
            double stddevDiff = std::abs(stddev1[0] - stddev2[0]) / 255.0;
            
            totalDiff += meanDiff + stddevDiff;
        }
        
        return 1.0 / (1.0 + totalDiff);
    }
    
    // LBP calculation
    static cv::Mat calculateLBP(const cv::Mat& gray) 
    {
        cv::Mat lbp = cv::Mat::zeros(gray.rows - 2, gray.cols - 2, CV_8UC1);
        
        for (int i = 1; i < gray.rows - 1; i++) {
            for (int j = 1; j < gray.cols - 1; j++) 
            {
                uchar center = gray.at<uchar>(i, j);
                unsigned char code = 0;
                
                code |= (gray.at<uchar>(i-1, j-1) >= center) << 7;
                code |= (gray.at<uchar>(i-1, j  ) >= center) << 6;
                code |= (gray.at<uchar>(i-1, j+1) >= center) << 5;
                code |= (gray.at<uchar>(i  , j+1) >= center) << 4;
                code |= (gray.at<uchar>(i+1, j+1) >= center) << 3;
                code |= (gray.at<uchar>(i+1, j  ) >= center) << 2;
                code |= (gray.at<uchar>(i+1, j-1) >= center) << 1;
                code |= (gray.at<uchar>(i  , j-1) >= center) << 0;
                
                lbp.at<uchar>(i-1, j-1) = code;
            }
        }
        
        return lbp;
    }
};


// PROGRESS BAR

class ProgressBar 
{
private:
    int total_;
    int current_;
    std::chrono::steady_clock::time_point start_;
    
public:
    ProgressBar(int total) : total_(total), current_(0) 
    {
        start_ = std::chrono::steady_clock::now();
    }
    
    void update(int current) {
        current_ = current;
        
        float progress = static_cast<float>(current) / total_;
        int barWidth = 50;
        int pos = static_cast<int>(barWidth * progress);
        
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_).count();
        double rate = current > 0 ? static_cast<double>(elapsed) / current : 0.0;
        int eta = rate > 0 ? static_cast<int>((total_ - current) * rate) : 0;
        
        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << "=";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% ";
        std::cout << "(" << current << "/" << total_ << ") ";
        
        if (current < total_ && eta > 0) {
            int h = eta / 3600;
            int m = (eta % 3600) / 60;
            int s = eta % 60;
            std::cout << "ETA: ";
            if (h > 0) std::cout << h << "h " << m << "m";
            else if (m > 0) std::cout << m << "m " << s << "s";
            else std::cout << s << "s";
        }
        std::cout << std::flush;
    }
    
    void finish() {
        update(total_);
        std::cout << "\n";
    }
};


// MATCH RESULT

struct MatchResult 
{
    std::string path;
    double score;
    
    bool operator>(const MatchResult& other) const 
    {
        return score > other.score;
    }
};


// HELPER FUNCTIONS


void printUsage(const char* program) 
{
    std::cout << "Usage: " << program << " -p <probe> -s <source> -d <dest> [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -p <path>    Probe image (face to search for)\n";
    std::cout << "  -s <path>    Source folder (images to search in)\n";
    std::cout << "  -d <path>    Destination folder (for matches)\n\n";
    std::cout << "Optional:\n";
    std::cout << "  -t <0.0-1.0> Similarity threshold (default: 0.65)\n";
    std::cout << "  -v           Verbose output\n";
    std::cout << "  -h           Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " -p face.jpg -s ./photos -d ./matches\n";
    std::cout << "  " << program << " -p face.jpg -s ./photos -d ./matches -t 0.70 -v\n\n";
}

Config parseArguments(int argc, char* argv[]) 
{
    Config config;
    
    for (int i = 1; i < argc; i++) 
    {
        std::string arg = argv[i];
        
        if (arg == "-p" && i + 1 < argc) 
        {
            config.probeImagePath = argv[++i];
        } else if (arg == "-s" && i + 1 < argc) 
        {
            config.sourceFolderPath = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) 
        {
            config.destFolderPath = argv[++i];
        } else if (arg == "-t" && i + 1 < argc) 
        {
            config.threshold = std::stof(argv[++i]);
        } else if (arg == "-v") 
        {
            config.verbose = true;
        } else if (arg == "-h") 
        {
            printUsage(argv[0]);
            exit(0);
        }
    }
    
    return config;
}

bool validateConfig(const Config& config) 
{
    if (config.probeImagePath.empty() || config.sourceFolderPath.empty() || config.destFolderPath.empty()) 
    {
        std::cerr << "Error: Missing required parameters\n";
        return false;
    }
    
    if (!PathUtils::fileExists(config.probeImagePath)) {
        std::cerr << "Error: Probe image not found: " << config.probeImagePath << "\n";
        return false;
    }
    
    if (!PathUtils::fileExists(config.sourceFolderPath)) {
        std::cerr << "Error: Source folder not found: " << config.sourceFolderPath << "\n";
        return false;
    }
    
    if (!ImageUtils::isValidImage(config.probeImagePath)) {
        std::cerr << "Error: Invalid probe image\n";
        return false;
    }
    
    if (config.threshold < 0.0f || config.threshold > 1.0f) {
        std::cerr << "Error: Threshold must be between 0.0 and 1.0\n";
        return false;
    }
    
    return true;
}

std::vector<std::string> collectImages(const std::string& folder) 
{
    std::vector<std::string> images;
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(folder)) 
        {
            if (!entry.is_regular_file()) continue;
            
            std::string path = entry.path().string();
            std::string ext = PathUtils::getFileExtension(path);
            
            if (PathUtils::isImageExtension(ext)) 
            {
                images.push_back(path);
            }
        }
        
        std::sort(images.begin(), images.end());
    } catch (const std::exception& e) {
        std::cerr << "Error scanning folder: " << e.what() << "\n";
    }
    
    return images;
}


// MAIN FUNCTION


int main(int argc, char* argv[]) 
{
    // Parse arguments
    Config config = parseArguments(argc, argv);
    
    if (!validateConfig(config)) 
    {
        printUsage(argv[0]);
        return 1;
    }
    
    // Print configuration
    std::cout << "\n Configuration:\n";
    std::cout << "   Probe:      " << config.probeImagePath << "\n";
    std::cout << "   Source:     " << config.sourceFolderPath << "\n";
    std::cout << "   Dest:       " << config.destFolderPath << "\n";
    std::cout << "   Threshold:  " << std::fixed << std::setprecision(2) << config.threshold << "\n\n";
    
    // Initialize detector
    FaceDetectorConfig detectorConfig;
    detectorConfig.detectionMethod = DetectionMethod::DNN_CAFFE;
    detectorConfig.detectLandmarks = false;
    detectorConfig.extractFaces = true;
    detectorConfig.alignFaces = true;
    detectorConfig.minConfidence = 0.5f;
    
    FaceDetector detector(detectorConfig);
    
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize detector\n";
        return 1;
    }
    
    // Extract probe face
    std::cout << "Detecting probe face...\n";
    auto probeFaces = detector.detectFaces(config.probeImagePath);
    
    if (probeFaces.empty() || probeFaces[0].faceImage.empty()) {
        std::cerr << "No face found in probe image\n";
        return 1;
    }
    
    cv::Mat probeFace = probeFaces[0].alignedFace.empty() ? 
                        probeFaces[0].faceImage : 
                        probeFaces[0].alignedFace;
    
    std::cout << "Probe face extracted (confidence: " 
              << std::fixed << std::setprecision(0) << (probeFaces[0].confidence * 100) << "%)\n\n";
    
    // Create destination folder
    try {
        std::filesystem::create_directories(config.destFolderPath);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create destination folder: " << e.what() << "\n";
        return 1;
    }
    
    // Collect source images
    auto sourceImages = collectImages(config.sourceFolderPath);
    
    if (sourceImages.empty()) {
        std::cerr << "No images found in source folder\n";
        return 1;
    }
    
    std::cout << "Found " << sourceImages.size() << " images to process\n\n";
    
    // Process images
    Statistics stats;
    stats.totalImages = sourceImages.size();
    stats.startTime = std::chrono::steady_clock::now();
    
    std::vector<MatchResult> matches;
    ProgressBar progress(stats.totalImages);
    
    for (const auto& imagePath : sourceImages) 
    {
        // Quick validation
        if (!ImageUtils::isValidImage(imagePath, false)) 
        {
            stats.skippedImages++;
            stats.processedImages++;
            progress.update(stats.processedImages);
            continue;
        }
        
        // Detect faces
        auto faces = detector.detectFaces(imagePath);
        
        if (faces.empty()) {
            stats.processedImages++;
            progress.update(stats.processedImages);
            continue;
        }
        
        stats.totalFaces += faces.size();
        
        // Compare each face
        bool foundMatch = false;
        double bestScore = 0.0;
        
        for (const auto& face : faces) 
        {
            if (face.faceImage.empty()) continue;
            
            cv::Mat targetFace = face.alignedFace.empty() ? face.faceImage : face.alignedFace;
            
            double score = EnhancedComparator::compareFacesAdvanced(probeFace, targetFace);
            
            if (score >= config.threshold) 
            {
                foundMatch = true;
                bestScore = std::max(bestScore, score);
                
                if (config.verbose) {
                    std::cout << "\n  Match: " << PathUtils::getFileName(imagePath) 
                              << " (score: " << std::fixed << std::setprecision(3) << score << ")\n";
                }
            }
        }
        
        if (foundMatch) 
        {
            matches.push_back({imagePath, bestScore});
            stats.matchedImages++;
        }
        
        stats.processedImages++;
        progress.update(stats.processedImages);
    }
    
    progress.finish();
    stats.endTime = std::chrono::steady_clock::now();
    
    // Copy matched images
    if (!matches.empty()) {
        std::cout << "\nCopying matched images...\n";
        
        // Sort by score (best first)
        std::sort(matches.begin(), matches.end(), std::greater<MatchResult>());
        
        int copied = 0;
        for (const auto& match : matches) 
        {
            try {
                std::filesystem::path src(match.path);
                std::filesystem::path dst = std::filesystem::path(config.destFolderPath) / src.filename();
                
                std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
                copied++;
            } catch (const std::exception& e) 
            {
                if (config.verbose) {
                    std::cerr << "  Failed to copy " << match.path << ": " << e.what() << "\n";
                }
            }
        }
        
        std::cout << "Copied " << copied << " image(s)\n";
    } else {
        std::cout << "\nNo matches found\n";
        std::cout << "Try lowering threshold: -t " << (config.threshold - 0.10f) << "\n";
    }
    
    // Print statistics
    stats.print();
    
    std::cout << "\nDone!\n\n";
    
    return 0;
}
