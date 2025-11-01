#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class FaceOrganizer 
{
public:
    void organizePerson(const std::string& referenceImage, const std::string& searchFolder);
    void setOutputFolder(const std::string& outputPath);
    void setSimilarityThreshold(double threshold);
    void enableGPU(bool enable);
    void setRecursiveSearch(bool recursive);
    int getMatchedCount() const;
    std::vector<std::string> getMatchedFiles() const;
    std::string getResultsFolder() const;

private:
    std::string outputFolder;
    double similarityThreshold = 0.7;
    bool useGPU = true;
    bool recursiveSearch = true;
    int matchedCount = 0;
    std::vector<std::string> matchedFiles;
};