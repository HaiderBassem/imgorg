#include "PathUtils.h"
#include "../Utils/Logger.h"

#include <algorithm>
#include<iostream>

std::string PathUtils::getFileName(const std::string &fullPath)
{
    std::filesystem::path path(fullPath);
    return path.filename().string();
}

std::string PathUtils::getFileExtension(const std::string &filePath)
{
    std::filesystem::path path(filePath);
    return path.extension().string();
}

std::string PathUtils::combinePaths(const std::string &path1, const std::string &path2)
{
    std::filesystem::path p1(path1);
    std::filesystem::path p2(path2);
    return (p1/p2).string();
}

bool PathUtils::isImageExtension(const std::string &exe)
{
    static const std::vector<std::string> imageExetension = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm"
    };

    std::string extLower = exe;
    std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);

    return std::find(imageExetension.begin(), imageExetension.end(), extLower) != imageExetension.end(); 
}

std::vector<std::string> PathUtils::getSupportedImageExetensions()
{
    return {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm"
    };
}

bool PathUtils::fileExists(const std::string &filePath)
{
    return std::filesystem::exists(filePath);
}

std::string PathUtils::getParentDirectory(const std::string &filePath)
{
    std::filesystem::path path(filePath);
    return path.parent_path().string();
}

bool PathUtils::createDirectoryIfNotExists(const std::string &path)
{
    try
    {
        return std::filesystem::create_directories(path);
    }
    catch(const std::exception& e)
    {
        std::cerr << "\033\31mFailed to create directory: " << e.what() << "\033[0m\n";
        Logger::instance().error("Failed to create directory:" + std::string(e.what()));
        return false;
    }
}

std::string PathUtils::getAbsolutePath(const std::string &relativePath)
{
    try
    {
        return std::filesystem::absolute(relativePath).string();
    }
    catch(const std::exception& e)
    {
        std::cerr << "\033[31m Failed to get absolute path " << e.what() << "\033[0m\n";
        Logger::instance().error("Failed to get absolute path: " + std::string(e.what()));
        return std::string();
    }
    
}
