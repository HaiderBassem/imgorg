#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

/**
 * Advanced Face Comparator using multiple algorithms
 * Combines 8 different comparison methods for maximum accuracy
 */
class FaceComparator {
public:
    FaceComparator();
    ~FaceComparator();
    
    // Main comparison methods
    double compareAdvanced(
        const cv::Mat& probeEmbedding,
        const std::vector<cv::Mat>& probeFeatures,
        const cv::Mat& targetFace
    );
    
    double compareFaces(const cv::Mat& face1, const cv::Mat& face2);
    
    // Feature extraction
    cv::Mat extractDeepFeatures(const cv::Mat& face);
    std::vector<cv::Mat> extractMultiScaleFeatures(const cv::Mat& face);
    
    // Individual comparison methods (public for testing)
    double compareHistogram(const cv::Mat& face1, const cv::Mat& face2);
    double compareSSIM(const cv::Mat& face1, const cv::Mat& face2);
    double compareLBP(const cv::Mat& face1, const cv::Mat& face2);
    double compareHOG(const cv::Mat& face1, const cv::Mat& face2);
    double compareORB(const cv::Mat& face1, const cv::Mat& face2);
    double compareSIFT(const cv::Mat& face1, const cv::Mat& face2);
    double compareColorMoments(const cv::Mat& face1, const cv::Mat& face2);
    double compareDeepEmbedding(const cv::Mat& emb1, const cv::Mat& emb2);
    
private:
    // Feature extraction helpers
    cv::Mat extractHistogramFeatures(const cv::Mat& face);
    cv::Mat extractLBPFeatures(const cv::Mat& face);
    cv::Mat extractHOGFeatures(const cv::Mat& face);
    cv::Mat extractColorMoments(const cv::Mat& face);
    cv::Mat calculateLBP(const cv::Mat& gray);
    
    // Comparison helpers
    double calculateHistogramSimilarity(const cv::Mat& hist1, const cv::Mat& hist2);
    double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2);
    double calculateCosineSimilarity(const cv::Mat& vec1, const cv::Mat& vec2);
    double calculateEuclideanDistance(const cv::Mat& vec1, const cv::Mat& vec2);
    
    // Feature detectors
    cv::Ptr<cv::ORB> orbDetector_;
    cv::Ptr<cv::SIFT> siftDetector_;
    cv::Ptr<cv::BFMatcher> orbMatcher_;
    cv::Ptr<cv::BFMatcher> siftMatcher_;
    cv::HOGDescriptor hogDescriptor_;
    
    // Preprocessing
    cv::Mat preprocessFace(const cv::Mat& face);
    std::vector<cv::Mat> createPyramid(const cv::Mat& face, int levels = 3);
};