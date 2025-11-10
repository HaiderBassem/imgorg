#pragma once 
#include <string>
#include <vector>
#include <filesystem>

/**
 * @namespace PathUtils
 * @brief Provides utility functions for file system path operations and image file handling.
 * 
 * This namespace contains cross-platform functions for path manipulation, file existence checks,
 * directory operations, and image file format validation. All functions are thread-safe.
 */
namespace PathUtils
{
    /**
     * @brief Extracts the filename from a full path string.
     * @param fullPath The complete file path including directory and filename.
     * @return std::string The filename with extension, or empty string if path is invalid.
     * 
     * @example
     * @code
     * std::string name = PathUtils::getFileName("/home/user/image.jpg"); // returns "image.jpg"
     * std::string name = PathUtils::getFileName("C:\\Users\\documents\\file.png"); // returns "file.png"
     * @endcode
     */
    std::string getFileName(const std::string& fullPath);

    /**
     * @brief Extracts the file extension from a file path.
     * @param filePath The file path to extract extension from.
     * @return std::string The file extension including the dot (e.g., ".jpg", ".png"), 
     *         or empty string if no extension found.
     * 
     * @example
     * @code
     * std::string ext = PathUtils::getFileExtension("photo.jpg"); // returns ".jpg"
     * std::string ext = PathUtils::getFileExtension("/path/to/image.png"); // returns ".png"
     * std::string ext = PathUtils::getFileExtension("file_without_extension"); // returns ""
     * @endcode
     */
    std::string getFileExtension(const std::string& filePath);

    /**
     * @brief Combines two path components into a single path.
     * @param path1 The first path component (typically a directory).
     * @param path2 The second path component (file or subdirectory).
     * @return std::string The combined path using system-specific path separator.
     * 
     * @note Automatically handles path separator differences between operating systems.
     * 
     * @example
     * @code
     * std::string full = PathUtils::combinePaths("/home/user", "images/photo.jpg");
     * // On Unix: returns "/home/user/images/photo.jpg"
     * // On Windows: returns "\\home\\user\\images\\photo.jpg"
     * @endcode
     */
    std::string combinePaths(const std::string& path1, const std::string& path2);

    /**
     * @brief Checks if a file extension is a supported image format.
     * @param exe The file extension to check (with or without leading dot).
     * @return true if the extension is a supported image format, false otherwise.
     * 
     * Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp, .gif, .ppm, .pgm, .pbm
     * 
     * @note Case-insensitive comparison. Both ".JPG" and ".jpg" are accepted.
     * 
     * @example
     * @code
     * bool isImage = PathUtils::isImageExtension(".jpg"); // returns true
     * bool isImage = PathUtils::isImageExtension("PNG"); // returns true
     * bool isImage = PathUtils::isImageExtension(".txt"); // returns false
     * @endcode
     */
    bool isImageExtension(const std::string& exe);

    /**
     * @brief Returns a list of all supported image file extensions.
     * @return std::vector<std::string> List of supported extensions including leading dots.
     * 
     * @example
     * @code
     * auto extensions = PathUtils::getSupportedImageExetensions();
     * // Returns: {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ...}
     * @endcode
     */
    std::vector<std::string> getSupportedImageExetensions();

    /**
     * @brief Checks if a file or directory exists at the specified path.
     * @param filePath The path to check for existence.
     * @return true if the file or directory exists, false otherwise.
     * 
     * @example
     * @code
     * bool exists = PathUtils::fileExists("/path/to/file.jpg"); // returns true if file exists
     * bool exists = PathUtils::fileExists("/nonexistent/path"); // returns false
     * @endcode
     */
    bool fileExists(const std::string& filePath);

    /**
     * @brief Extracts the parent directory path from a file path.
     * @param filePath The complete file path.
     * @return std::string The parent directory path, or empty string if no parent directory.
     * 
     * @example
     * @code
     * std::string parent = PathUtils::getParentDirectory("/home/user/images/photo.jpg");
     * // returns "/home/user/images"
     * @endcode
     */
    std::string getParentDirectory(const std::string& filePath);
    
    /**
     * @brief Creates a directory and all necessary parent directories if they don't exist.
     * @param path The directory path to create.
     * @return true if directory was created or already exists, false if creation failed.
     * 
     * @note This function creates all intermediate directories in the path if needed.
     * @note Logs errors to both console and logger if directory creation fails.
     * 
     * @example
     * @code
     * bool success = PathUtils::createDirectoryIfNotExists("/home/user/new_images");
     * // Creates /home, /home/user, and /home/user/new_images if they don't exist
     * @endcode
     */
    bool createDirectoryIfNotExists(const std::string& path);

    /**
     * @brief Converts a relative path to an absolute path.
     * @param relativePath The relative path to convert.
     * @return std::string The absolute path, or empty string if conversion fails.
     * 
     * @note Resolves symbolic links and normalizes the path.
     * @note Logs errors to both console and logger if conversion fails.
     * 
     * @example
     * @code
     * std::string absPath = PathUtils::getAbsolutePath("../images/photo.jpg");
     * // Might return "/home/user/project/images/photo.jpg" depending on current directory
     * @endcode
     */
    std::string getAbsolutePath(const std::string& relativePath);
} // namespace PathUtils