// #include "FaceOrganizer.h"
// #include "../FaceRecognition/FaceDetector.h"
// #include "../Utils/ImageUtils.h"
// #include "../Utils/ProgressTracker.h"
// #include "../FileSystem/PathUtils.h"

// #include <filesystem>
// #include <algorithm>
// #include <thread>
// #include <mutex>
// #include <atomic>
// #include <queue>
// #include <iostream>
// #include <iomanip>

// // ============================================================================
// // ORGANIZER STATISTICS IMPLEMENTATION
// // ============================================================================
// void OrganizerStatistics::print() const {
//     std::cout << "\n╔══════════════════════════════════════════════════════╗" << std::endl;
//     std::cout << "║                  STATISTICS REPORT                    ║" << std::endl;
//     std::cout << "╚══════════════════════════════════════════════════════╝" << std::endl;
//     std::cout << "📊 Processing Statistics:" << std::endl;
//     std::cout << "   - Total images processed: " << totalImagesProcessed << std::endl;
//     std::cout << "   - Total faces detected: " << totalFacesDetected << std::endl;
//     std::cout << "   - Total matches found: " << totalMatches << std::endl;
//     std::cout << "\n💾 Cache Statistics:" << std::endl;
//     std::cout << "   - Cache hits: " << cacheHits << std::endl;
//     std::cout << "   - Cache misses: " << cacheMisses << std::endl;
//     if (cacheHits + cacheMisses > 0) {
//         double hitRate = (cacheHits * 100.0) / (cacheHits + cacheMisses);
//         std::cout << "   - Hit rate: " << std::fixed << std::setprecision(1)
//                   << hitRate << "%" << std::endl;
//     }
//     std::cout << "\n⏱️  Performance:" << std::endl;
//     std::cout << "   - Total time: " << std::fixed << std::setprecision(2)
//               << totalProcessingTime << "s" << std::endl;
//     if (totalImagesProcessed > 0) {
//         std::cout << "   - Average per image: " << std::fixed << std::setprecision(3)
//                   << (totalProcessingTime / totalImagesProcessed) << "s" << std::endl;
//         std::cout << "   - Throughput: " << std::fixed << std::setprecision(1)
//                   << (totalImagesProcessed / totalProcessingTime) << " images/s" << std::endl;
//     }
//     std::cout << std::string(56, '═') << std::endl;
// }

// // ============================================================================
// // FACE CACHE IMPLEMENTATION
// // ============================================================================
// class FaceCache::Impl {
// public:
//     std::unordered_map<std::string, FaceCacheEntry> cache_;
//     mutable std::mutex mutex_;
//     size_t maxSize_;
//     std::atomic<size_t> hits_{0};
//     std::atomic<size_t> misses_{0};
    
//     explicit Impl(size_t maxSize) : maxSize_(maxSize) {}
    
//     void evictOldest() {
//         if (cache_.empty()) return;
        
//         auto oldest = cache_.begin();
//         for (auto it = cache_.begin(); it != cache_.end(); ++it) {
//             if (it->second.timestamp < oldest->second.timestamp) {
//                 oldest = it;
//             }
//         }
//         cache_.erase(oldest);
//     }
// };

// FaceCache::FaceCache(size_t maxSize)
//     : pImpl_(std::make_unique<Impl>(maxSize)) {}

// FaceCache::~FaceCache() = default;

// void FaceCache::put(const std::string& path, const FaceCacheEntry& entry) {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
    
//     if (pImpl_->cache_.size() >= pImpl_->maxSize_) {
//         pImpl_->evictOldest();
//     }
    
//     pImpl_->cache_[path] = entry;
// }

// bool FaceCache::get(const std::string& path, FaceCacheEntry& entry) {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
    
//     auto it = pImpl_->cache_.find(path);
//     if (it != pImpl_->cache_.end()) {
//         entry = it->second;
//         pImpl_->hits_++;
//         return true;
//     }
    
//     pImpl_->misses_++;
//     return false;
// }

// bool FaceCache::contains(const std::string& path) const {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
//     return pImpl_->cache_.find(path) != pImpl_->cache_.end();
// }

// void FaceCache::remove(const std::string& path) {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
//     pImpl_->cache_.erase(path);
// }

// void FaceCache::clear() {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
//     pImpl_->cache_.clear();
// }

// size_t FaceCache::size() const {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
//     return pImpl_->cache_.size();
// }

// size_t FaceCache::capacity() const {
//     return pImpl_->maxSize_;
// }

// void FaceCache::setCapacity(size_t maxSize) {
//     std::lock_guard<std::mutex> lock(pImpl_->mutex_);
//     pImpl_->maxSize_ = maxSize;
    
//     while (pImpl_->cache_.size() > maxSize) {
//         pImpl_->evictOldest();
//     }
// }

// size_t FaceCache::getHits() const {
//     return pImpl_->hits_.load();
// }

// size_t FaceCache::getMisses() const {
//     return pImpl_->misses_.load();
// }

// double FaceCache::getHitRate() const {
//     size_t hits = pImpl_->hits_.load();
//     size_t misses = pImpl_->misses_.load();
//     size_t total = hits + misses;
//     return total > 0 ? (hits * 100.0 / total) : 0.0;
// }

// // ============================================================================
// // ENHANCED FACE COMPARATOR IMPLEMENTATION
// // ============================================================================
// class EnhancedFaceComparator::Impl {
// public:
//     cv::Ptr<cv::ORB> orb_;
//     cv::Ptr<cv::BFMatcher> matcher_;
    
//     Impl() {
//         orb_ = cv::ORB::create(500);
//         matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING);
//     }
    
//     cv::Mat extractColorHistogram(const cv::Mat& face) {
//         cv::Mat lab;
//         cv::cvtColor(face, lab, cv::COLOR_BGR2Lab);
        
//         cv::Mat hist;
//         int histSize[] = {8, 8, 8};
//         float lRanges[] = {0, 256};
//         float aRanges[] = {0, 256};
//         float bRanges[] = {0, 256};
//         const float* ranges[] = {lRanges, aRanges, bRanges};
//         int channels[] = {0, 1, 2};
        
//         cv::calcHist(&lab, 1, channels, cv::Mat(), hist, 3, histSize, ranges);
//         cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
        
//         return hist.reshape(1, 1);
//     }
    
//     cv::Mat extractLBP(const cv::Mat& face) {
//         cv::Mat gray;
//         cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        
//         cv::Mat lbp = cv::Mat::zeros(gray.rows - 2, gray.cols - 2, CV_8UC1);
        
//         for (int i = 1; i < gray.rows - 1; i++) {
//             for (int j = 1; j < gray.cols - 1; j++) {
//                 uchar center = gray.at<uchar>(i, j);
//                 unsigned char code = 0;
//                 code |= (gray.at<uchar>(i-1,j-1) > center) << 7;
//                 code |= (gray.at<uchar>(i-1,j) > center) << 6;
//                 code |= (gray.at<uchar>(i-1,j+1) > center) << 5;
//                 code |= (gray.at<uchar>(i,j+1) > center) << 4;
//                 code |= (gray.at<uchar>(i+1,j+1) > center) << 3;
//                 code |= (gray.at<uchar>(i+1,j) > center) << 2;
//                 code |= (gray.at<uchar>(i+1,j-1) > center) << 1;
//                 code |= (gray.at<uchar>(i,j-1) > center) << 0;
//                 lbp.at<uchar>(i-1, j-1) = code;
//             }
//         }
        
//         cv::Mat lbpHist;
//         int histSize = 256;
//         float range[] = {0, 256};
//         const float* ranges[] = {range};
//         cv::calcHist(&lbp, 1, 0, cv::Mat(), lbpHist, 1, &histSize, ranges);
//         cv::normalize(lbpHist, lbpHist, 0, 1, cv::NORM_MINMAX);
        
//         return lbpHist.reshape(1, 1);
//     }
    
//     cv::Mat extractHOG(const cv::Mat& face) {
//         cv::Mat gray;
//         cv::cvtColor(face, gray, cv::COLOR_BGR2GRAY);
        
//         cv::HOGDescriptor hog(cv::Size(160, 160), cv::Size(16, 16),
//                              cv::Size(8, 8), cv::Size(8, 8), 9);
        
//         std::vector<float> descriptors;
//         hog.compute(gray, descriptors);
        
//         cv::Mat hogMat(descriptors);
//         cv::normalize(hogMat, hogMat, 0, 1, cv::NORM_MINMAX);
        
//         return hogMat.reshape(1, 1);
//     }
// };

// EnhancedFaceComparator::EnhancedFaceComparator()
//     : pImpl_(std::make_unique<Impl>()) {}

// EnhancedFaceComparator::~EnhancedFaceComparator() = default;

// cv::Mat EnhancedFaceComparator::extractFeatures(const cv::Mat& face) {
//     if (face.empty()) return cv::Mat();
    
//     cv::Mat resized;
//     cv::resize(face, resized, cv::Size(160, 160));
    
//     std::vector<cv::Mat> features;
//     features.push_back(pImpl_->extractColorHistogram(resized));
//     features.push_back(pImpl_->extractLBP(resized));
//     features.push_back(pImpl_->extractHOG(resized));
    
//     cv::Mat combined;
//     cv::hconcat(features, combined);
    
//     return combined;
// }

// double EnhancedFaceComparator::compareHistogram(const cv::Mat& face1, const cv::Mat& face2) {
//     cv::Mat hist1 = pImpl_->extractColorHistogram(face1);
//     cv::Mat hist2 = pImpl_->extractColorHistogram(face2);
    
//     double corr = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
//     return (corr + 1.0) / 2.0;
// }

// double EnhancedFaceComparator::compareSSIM(const cv::Mat& face1, const cv::Mat& face2) {
//     cv::Mat gray1, gray2;
//     cv::cvtColor(face1, gray1, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(face2, gray2, cv::COLOR_BGR2GRAY);
    
//     cv::resize(gray1, gray1, cv::Size(160, 160));
//     cv::resize(gray2, gray2, cv::Size(160, 160));
    
//     cv::Mat I1, I2;
//     gray1.convertTo(I1, CV_32F);
//     gray2.convertTo(I2, CV_32F);
    
//     const double C1 = 6.5025, C2 = 58.5225;
    
//     cv::Mat mu1, mu2;
//     cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
//     cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    
//     cv::Mat mu1_2 = mu1.mul(mu1);
//     cv::Mat mu2_2 = mu2.mul(mu2);
//     cv::Mat mu1_mu2 = mu1.mul(mu2);
    
//     cv::Mat sigma1_2, sigma2_2, sigma12;
//     cv::GaussianBlur(I1.mul(I1), sigma1_2, cv::Size(11, 11), 1.5);
//     sigma1_2 -= mu1_2;
    
//     cv::GaussianBlur(I2.mul(I2), sigma2_2, cv::Size(11, 11), 1.5);
//     sigma2_2 -= mu2_2;
    
//     cv::GaussianBlur(I1.mul(I2), sigma12, cv::Size(11, 11), 1.5);
//     sigma12 -= mu1_mu2;
    
//     cv::Mat t1 = 2 * mu1_mu2 + C1;
//     cv::Mat t2 = 2 * sigma12 + C2;
//     cv::Mat numerator = t1.mul(t2);
    
//     t1 = mu1_2 + mu2_2 + C1;
//     t2 = sigma1_2 + sigma2_2 + C2;
//     cv::Mat denominator = t1.mul(t2);
    
//     cv::Mat ssim_map;
//     cv::divide(numerator, denominator, ssim_map);
    
//     cv::Scalar meanSSIM = cv::mean(ssim_map);
//     return meanSSIM[0];
// }

// double EnhancedFaceComparator::compareFeatures(const cv::Mat& face1, const cv::Mat& face2) {
//     cv::Mat feat1 = extractFeatures(face1);
//     cv::Mat feat2 = extractFeatures(face2);
    
//     if (feat1.empty() || feat2.empty()) return 0.0;
    
//     double corr = cv::compareHist(feat1, feat2, cv::HISTCMP_CORREL);
//     return (corr + 1.0) / 2.0;
// }

// double EnhancedFaceComparator::compareEmbeddings(const cv::Mat& face1, const cv::Mat& face2) {
//     cv::Mat lbp1 = pImpl_->extractLBP(face1);
//     cv::Mat lbp2 = pImpl_->extractLBP(face2);
    
//     double corr = cv::compareHist(lbp1, lbp2, cv::HISTCMP_CORREL);
//     return (corr + 1.0) / 2.0;
// }

// double EnhancedFaceComparator::compareKeypoints(const cv::Mat& face1, const cv::Mat& face2) {
//     cv::Mat gray1, gray2;
//     cv::cvtColor(face1, gray1, cv::COLOR_BGR2GRAY);
//     cv::cvtColor(face2, gray2, cv::COLOR_BGR2GRAY);
    
//     cv::resize(gray1, gray1, cv::Size(200, 200));
//     cv::resize(gray2, gray2, cv::Size(200, 200));
    
//     std::vector<cv::KeyPoint> kp1, kp2;
//     cv::Mat desc1, desc2;
    
//     pImpl_->orb_->detectAndCompute(gray1, cv::noArray(), kp1, desc1);
//     pImpl_->orb_->detectAndCompute(gray2, cv::noArray(), kp2, desc2);
    
//     if (desc1.empty() || desc2.empty()) return 0.0;
    
//     std::vector<std::vector<cv::DMatch>> matches;
//     pImpl_->matcher_->knnMatch(desc1, desc2, matches, 2);
    
//     std::vector<cv::DMatch> goodMatches;
//     for (const auto& m : matches) {
//         if (m.size() >= 2 && m[0].distance < 0.75 * m[1].distance) {
//             goodMatches.push_back(m[0]);
//         }
//     }
    
//     double matchRatio = static_cast<double>(goodMatches.size()) /
//                        std::min(kp1.size(), kp2.size());
    
//     return std::min(1.0, matchRatio);
// }

// double EnhancedFaceComparator::compareFaces(const cv::Mat& face1, const cv::Mat& face2, bool useDeep) {
//     if (face1.empty() || face2.empty()) return 0.0;
    
//     cv::Mat resized1, resized2;
//     cv::resize(face1, resized1, cv::Size(160, 160));
//     cv::resize(face2, resized2, cv::Size(160, 160));
    
//     std::vector<double> scores;
    
//     // Method 1: Feature-based (fast & accurate)
//     scores.push_back(compareFeatures(resized1, resized2));
    
//     // Method 2: Histogram
//     scores.push_back(compareHistogram(resized1, resized2));
    
//     if (useDeep) {
//         // Method 3: SSIM
//         scores.push_back(compareSSIM(resized1, resized2));
        
//         // Method 4: LBP Embeddings
//         scores.push_back(compareEmbeddings(resized1, resized2));
        
//         // Method 5: Keypoint matching
//         scores.push_back(compareKeypoints(resized1, resized2));
        
//         // Weighted average (5 methods)
//         return scores[0] * 0.30 +  // Features
//                scores[1] * 0.20 +  // Histogram
//                scores[2] * 0.25 +  // SSIM
//                scores[3] * 0.15 +  // Embeddings
//                scores[4] * 0.10;   // Keypoints
//     } else {
//         // Fast mode (2 methods only)
//         return scores[0] * 0.6 + scores[1] * 0.4;
//     }
// }

// // Continue in next part...
