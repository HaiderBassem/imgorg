#pragma once 
#include <string>
#include <vector>
#include <filesystem>

namespace PathUtils
{
    std::string getFileName(const std::string& fullPath);
    std::string getFileExetension(const std::string& filePath);
    // combine tow paths 
    std::string combinePaths(const std::string& path1, const std::string& path2);

    bool isImageExetension(const std::string& exe);

    std::vector<std::string> getSupportedImageExetensions();

    bool fileExists(const std::string& filePath);

    std::string getParentDirectory(const std::string& filePath);
    
    bool createDirectoryIfNotExists(const std::string& path);

    std::string getAbsolutePath(const std::string& relativePath);
} // namespace PathUtils
