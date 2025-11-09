#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

class ProgressBar {
public:
    ProgressBar(int total, const std::string& label = "Progress")
        : total_(total)
        , current_(0)
        , label_(label)
        , startTime_(std::chrono::steady_clock::now())
    {
        if (total_ <= 0) total_ = 1;
    }
    
    void update(int current) {
        current_ = current;
        display();
    }
    
    void finish() {
        current_ = total_;
        display();
        std::cout << "\n";
    }
    
private:
    void display() {
        int barWidth = 50;
        float progress = static_cast<float>(current_) / total_;
        int pos = static_cast<int>(barWidth * progress);
        
        // Calculate ETA
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime_).count();
        double rate = current_ > 0 ? static_cast<double>(elapsed) / current_ : 0.0;
        int eta = rate > 0 ? static_cast<int>((total_ - current_) * rate) : 0;
        
        // Print progress bar
        std::cout << "\r" << label_ << ": [";
        
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "█";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1)
                  << (progress * 100.0) << "% "
                  << "(" << current_ << "/" << total_ << ") ";
        
        if (current_ < total_ && eta > 0) {
            std::cout << "ETA: " << formatTime(eta);
        } else if (current_ == total_) {
            std::cout << "Done in " << formatTime(elapsed) << "   ";
        }
        
        std::cout << std::flush;
    }
    
    std::string formatTime(int seconds) {
        int h = seconds / 3600;
        int m = (seconds % 3600) / 60;
        int s = seconds % 60;
        
        if (h > 0) {
            return std::to_string(h) + "h " + std::to_string(m) + "m";
        } else if (m > 0) {
            return std::to_string(m) + "m " + std::to_string(s) + "s";
        } else {
            return std::to_string(s) + "s";
        }
    }
    
    int total_;
    int current_;
    std::string label_;
    std::chrono::steady_clock::time_point startTime_;
};