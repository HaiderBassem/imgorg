#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <opencv2/opencv.hpp>

// Forward declarations
class FaceDetector;
class ProgressTracker;

// ============================================================================
// CONFIGURATION STRUCTURE
// ============================================================================
struct FaceOrganizerConfig {
    // Detection settings
    float similarityThreshold = 0.65f;      // 65% similarity (0.0 - 1.0)
    float minFaceConfidence = 0.6f;         // Minimum face detection confidence
    int minFaceSize = 80;                   // Minimum face size in pixels
    
    // Performance settings
    int numThreads = -1;                    // -1 = auto-detect hardware threads
    int batchSize = 50;                     // Process images in batches
    bool useCache = true;                   // Cache face embeddings
    bool verboseOutput = true;              // Show detailed progress
    
    // Quality settings
    bool skipBlurryFaces = true;            // Skip low quality faces
    float minSharpness = 30.0f;             // Minimum sharpness threshold
    bool alignFaces = true;                 // Align faces before comparison
    
    // Advanced settings
    bool useMultipleDetectors = true;       // Try multiple detection methods
    bool saveDebugImages = false;           // Save face crops for debugging
    int maxFacesPerImage = 10;              // Max faces to process per image
    
    // Comparison settings
    bool useDeepComparison = true;          // Use all 5 comparison methods
    bool useFastMode = false;               // Use only 2 fast methods
};

// ============================================================================
// FACE CACHE ENTRY
// ============================================================================
struct FaceCacheEntry {
    std::string imagePath;
    std::vector<cv::Mat> faceEmbeddings;    // Face features for comparison
    std::vector<cv::Rect> faceBounds;       // Face bounding boxes
    std::vector<float> faceQualities;       // Quality scores
    std::chrono::system_clock::time_point timestamp;
    
    FaceCacheEntry() = default;
    
    bool isValid() const {
        return !faceEmbeddings.empty() && !imagePath.empty();
    }
};

// ============================================================================
// MATCH RESULT STRUCTURE
// ============================================================================
struct MatchResult {
    std::string imagePath;
    double similarityScore;
    int faceIndex;
    cv::Rect faceBound;
    
    MatchResult() : similarityScore(0.0), faceIndex(-1) {}
    MatchResult(const std::string& path, double score, int idx = 0, cv::Rect bound = cv::Rect())
        : imagePath(path), similarityScore(score), faceIndex(idx), faceBound(bound) {}
    
    bool operator>(const MatchResult& other) const {
        return similarityScore > other.similarityScore;
    }
};

// ============================================================================
// ORGANIZER STATISTICS
// ============================================================================
struct OrganizerStatistics {
    int totalImagesProcessed = 0;
    int totalFacesDetected = 0;
    int totalMatches = 0;
    int cacheHits = 0;
    int cacheMisses = 0;
    double averageProcessingTime = 0.0;
    double totalProcessingTime = 0.0;
    
    void reset() {
        totalImagesProcessed = 0;
        totalFacesDetected = 0;
        totalMatches = 0;
        cacheHits = 0;
        cacheMisses = 0;
        averageProcessingTime = 0.0;
        totalProcessingTime = 0.0;
    }
    
    void print() const;
};

// ============================================================================
// PROGRESS CALLBACK
// ============================================================================
using ProgressCallback = std::function<void(int current, int total, const std::string& message)>;

// ============================================================================
// FACE ORGANIZER CLASS
// ============================================================================
class FaceOrganizer {
public:
    // Constructor & Destructor
    FaceOrganizer();
    explicit FaceOrganizer(const FaceOrganizerConfig& config);
    ~FaceOrganizer();
    
    // Initialization
    bool initialize();
    bool initialize(const FaceOrganizerConfig& config);
    bool isInitialized() const;
    
    // Configuration
    void setConfig(const FaceOrganizerConfig& config);
    FaceOrganizerConfig getConfig() const;
    
    // Main operations
    bool organizeFaces(const std::string& probeImagePath,
                       const std::string& srcFolder,
                       const std::string& dstFolder);
    
    bool organizeFacesWithProgress(const std::string& probeImagePath,
                                   const std::string& srcFolder,
                                   const std::string& dstFolder,
                                   ProgressCallback callback);
    
    // Search operations
    std::vector<MatchResult> findSimilarFaces(const std::string& probeImagePath,
                                              const std::string& srcFolder);
    
    std::vector<MatchResult> findSimilarFacesInImages(const cv::Mat& probeFace,
                                                      const std::vector<std::string>& imagePaths);
    
    // Batch operations
    bool organizeFacesBatch(const std::vector<std::string>& probeImages,
                           const std::string& srcFolder,
                           const std::string& baseDstFolder);
    
    // Face comparison
    double compareFaces(const cv::Mat& face1, const cv::Mat& face2);
    double compareFacesDetailed(const cv::Mat& face1, const cv::Mat& face2,
                               std::vector<double>& methodScores);
    
    // Cache management
    void clearCache();
    size_t getCacheSize() const;
    void setCacheMaxSize(size_t maxSize);
    
    // Statistics
    OrganizerStatistics getStatistics() const;
    void resetStatistics();
    void printStatistics() const;
    
    // Utility
    std::vector<std::string> getSupportedImageFormats() const;
    bool validatePaths(const std::string& probeImagePath,
                      const std::string& srcFolder,
                      const std::string& dstFolder) const;
    
    // Error handling
    std::string getLastError() const;
    bool hasError() const;
    
private:
    // Internal processing methods
    bool processFaceImage(const std::string& imagePath, FaceCacheEntry& entry);
    double findBestMatch(const cv::Mat& probeFace, const FaceCacheEntry& entry);
    
    bool extractProbeFace(const std::string& probeImagePath, cv::Mat& probeFace);
    std::vector<std::string> collectImageFiles(const std::string& folder);
    
    void processImagesParallel(const std::vector<std::string>& imageFiles,
                               const cv::Mat& probeFace,
                               std::vector<MatchResult>& matches,
                               ProgressCallback callback);
    
    bool copyMatchingImages(const std::vector<MatchResult>& matches,
                           const std::string& dstFolder);
    
    // Feature extraction
    cv::Mat extractFaceFeatures(const cv::Mat& face);
    cv::Mat extractColorHistogram(const cv::Mat& face);
    cv::Mat extractLBPFeatures(const cv::Mat& face);
    cv::Mat extractHOGFeatures(const cv::Mat& face);
    
    // Face quality assessment
    bool isValidFace(const cv::Mat& face, float& qualityScore);
    
    // Thread management
    void initializeThreadPool();
    void shutdownThreadPool();
    
    // Member variables
    class Impl;
    std::unique_ptr<Impl> pImpl_;
    
    FaceOrganizerConfig config_;
    bool initialized_;
    std::string lastError_;
    OrganizerStatistics stats_;
    mutable std::mutex mutex_;
};

// ============================================================================
// ENHANCED FACE COMPARATOR (Helper Class)
// ============================================================================
class EnhancedFaceComparator {
public:
    EnhancedFaceComparator();
    ~EnhancedFaceComparator();
    
    // Compare faces with multiple methods
    double compareFaces(const cv::Mat& face1, const cv::Mat& face2, bool useDeep = true);
    
    // Get individual method scores
    double compareHistogram(const cv::Mat& face1, const cv::Mat& face2);
    double compareSSIM(const cv::Mat& face1, const cv::Mat& face2);
    double compareFeatures(const cv::Mat& face1, const cv::Mat& face2);
    double compareEmbeddings(const cv::Mat& face1, const cv::Mat& face2);
    double compareKeypoints(const cv::Mat& face1, const cv::Mat& face2);
    
    // Feature extraction
    cv::Mat extractFeatures(const cv::Mat& face);
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

// ============================================================================
// THREAD-SAFE FACE CACHE (Helper Class)
// ============================================================================
class FaceCache {
public:
    explicit FaceCache(size_t maxSize = 1000);
    ~FaceCache();
    
    // Cache operations
    void put(const std::string& path, const FaceCacheEntry& entry);
    bool get(const std::string& path, FaceCacheEntry& entry);
    bool contains(const std::string& path) const;
    
    void remove(const std::string& path);
    void clear();
    
    // Cache info
    size_t size() const;
    size_t capacity() const;
    void setCapacity(size_t maxSize);
    
    // Statistics
    size_t getHits() const;
    size_t getMisses() const;
    double getHitRate() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};