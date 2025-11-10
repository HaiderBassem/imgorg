#include "PathUtils.h"
#include "../Utils/Logger.h"
#include <algorithm>
#include <iostream>

/**
 * @brief Extracts the filename from a full path string.
 * 
 * Uses std::filesystem::path to reliably extract the filename component
 * regardless of the operating system's path separator.
 * 
 * @param fullPath The complete file path including directory and filename.
 * @return std::string The filename with extension.
 * 
 * @implementation
 * - Creates a filesystem path object from the input string
 * - Extracts the filename component using filename()
 * - Converts to string and returns
 */
std::string PathUtils::getFileName(const std::string &fullPath)
{
    std::filesystem::path path(fullPath);
    return path.filename().string();
}

/**
 * @brief Extracts the file extension from a file path.
 * 
 * @param filePath The file path to extract extension from.
 * @return std::string The file extension including the dot.
 * 
 * @implementation
 * - Creates a filesystem path object
 * - Uses extension() method to get the extension
 * - Returns the extension as string including the dot
 */
std::string PathUtils::getFileExtension(const std::string &filePath)
{
    std::filesystem::path path(filePath);
    return path.extension().string();
}

/**
 * @brief Combines two path components into a single path.
 * 
 * Uses the std::filesystem::path operator/ which automatically handles
 * the correct path separator for the current operating system.
 * 
 * @param path1 The first path component.
 * @param path2 The second path component.
 * @return std::string The combined path.
 * 
 * @implementation
 * - Creates two filesystem path objects
 * - Uses operator/ to combine them (handles separators automatically)
 * - Converts back to string and returns
 */
std::string PathUtils::combinePaths(const std::string &path1, const std::string &path2)
{
    std::filesystem::path p1(path1);
    std::filesystem::path p2(path2);
    return (p1 / p2).string();
}

/**
 * @brief Checks if a file extension is a supported image format.
 * 
 * Performs case-insensitive comparison against a predefined list
 * of supported image extensions.
 * 
 * @param exe The file extension to check.
 * @return bool True if supported image format.
 * 
 * @implementation
 * - Defines static list of supported extensions for efficiency
 * - Converts input to lowercase for case-insensitive comparison
 * - Uses std::find to check if extension exists in supported list
 */
bool PathUtils::isImageExtension(const std::string &exe)
{
    // Static for efficiency - initialized once
    static const std::vector<std::string> imageExetension = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm"
    };

    // Convert to lowercase for case-insensitive comparison
    std::string extLower = exe;
    std::transform(extLower.begin(), extLower.end(), extLower.begin(), ::tolower);

    // Check if extension exists in supported list
    return std::find(imageExetension.begin(), imageExetension.end(), extLower) != imageExetension.end(); 
}

/**
 * @brief Returns a list of all supported image file extensions.
 * 
 * @return std::vector<std::string> List of supported extensions.
 * 
 * @implementation
 * - Returns a vector with all supported image extensions
 * - Includes both common and less common image formats
 * - Extensions include leading dots
 */
std::vector<std::string> PathUtils::getSupportedImageExetensions()
{
    return {
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
        ".webp", ".gif", ".ppm", ".pgm", ".pbm"
    };
}

/**
 * @brief Checks if a file or directory exists at the specified path.
 * 
 * @param filePath The path to check for existence.
 * @return bool True if path exists.
 * 
 * @implementation
 * - Uses std::filesystem::exists to check path existence
 * - Works for both files and directories
 * - Returns immediately with result
 */
bool PathUtils::fileExists(const std::string &filePath)
{
    return std::filesystem::exists(filePath);
}

/**
 * @brief Extracts the parent directory path from a file path.
 * 
 * @param filePath The complete file path.
 * @return std::string The parent directory path.
 * 
 * @implementation
 * - Creates filesystem path object
 * - Uses parent_path() to get parent directory
 * - Returns as string
 */
std::string PathUtils::getParentDirectory(const std::string &filePath)
{
    std::filesystem::path path(filePath);
    return path.parent_path().string();
}

/**
 * @brief Creates a directory and all necessary parent directories if they don't exist.
 * 
 * Uses std::filesystem::create_directories which creates all intermediate
 * directories in the path if they don't exist.
 * 
 * @param path The directory path to create.
 * @return bool True if directory was created or already exists.
 * 
 * @implementation
 * - First checks if path already exists (optimization)
 * - Uses create_directories to create path with all parents
 * - Handles exceptions and logs errors appropriately
 * - Returns success status
 */
bool PathUtils::createDirectoryIfNotExists(const std::string &path)
{
    try
    {
        // Check if directory already exists (avoid unnecessary creation)
        if (std::filesystem::exists(path)) {
            return true;  
        }
        
        // Create directory and all parent directories
        return std::filesystem::create_directories(path);
    }
    catch(const std::exception& e)
    {
        // Log error to both console and logger
        std::cerr << "\033[31mFailed to create directory: " << e.what() << "\033[0m\n";
        Logger::instance().error("Failed to create directory: " + std::string(e.what()));
        return false;
    }
}

/**
 * @brief Converts a relative path to an absolute path.
 * 
 * Resolves the absolute path based on the current working directory.
 * Handles path normalization and symbolic link resolution.
 * 
 * @param relativePath The relative path to convert.
 * @return std::string The absolute path.
 * 
 * @implementation
 * - Uses std::filesystem::absolute for conversion
 * - Wraps in try-catch for error handling
 * - Logs errors to both console and logger
 * - Returns empty string on failure
 */
std::string PathUtils::getAbsolutePath(const std::string &relativePath)
{
    try
    {
        return std::filesystem::absolute(relativePath).string();
    }
    catch(const std::exception& e)
    {
        // Log error with colored console output and logger
        std::cerr << "\033[31m Failed to get absolute path " << e.what() << "\033[0m\n";
        Logger::instance().error("Failed to get absolute path: " + std::string(e.what()));
        return std::string();
    }
}