#include "Logger.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <regex>
#include <filesystem>
#include <cstdlib> 

Logger& Logger::instance(const std::string& filename)
{
    static Logger instance(filename);
    return instance;
}

Logger::Logger(const std::string& filename) : _stopRequested(false)
{
    std::string basePath;
   #if defined(_WIN32)
        const char* userProfile = std::getenv("USERPROFILE");
        if(userProfile)
            basePath = std::string(userProfile) + "\\Documents\\imgorg\\";
        else
            basePath = ".\\";    
    #else
        const char* home =std::getenv("HOME");
        if(home)
            basePath = std::string(home) + "/Documents/imgorg/";
        else
            basePath = "./";
    #endif            
// if the Documents folder not exists, create one...
try{
    std::filesystem::create_directories(basePath);
}catch(...)
{
    std::cerr << "[Logger Warning] Failed to create Documents folder. \n";
}

std::string fullPath = basePath + filename;

_logFile.open(fullPath, std::ios::app);
if(!_logFile.is_open())
    std::cerr << "\033[33m [WARNRING] Failed to open log file: " << fullPath << "\033[0m \n";
else
    std::cout << "\033[32 [Logger] Writing to: " << fullPath << "\033[0m" <<std::endl; 

_workerThread = std::thread(&Logger::workerThread, this);
}


Logger::~Logger()
{
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _stopRequested = true;
    }

    _condition.notify_all();
    if (_workerThread.joinable())
        _workerThread.join();

    if (_logFile.is_open()) {
        _logFile.flush();
        _logFile.close();
    }
}

void Logger::workerThread()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(_queueMutex);
        _condition.wait(lock, [this]() { 
            return !_messageQueue.empty() || _stopRequested; 
        });

        if (_stopRequested && _messageQueue.empty())
            break;

        std::queue<std::string> messages;
        std::swap(_messageQueue, messages);
        lock.unlock();

        while (!messages.empty())
        {
            const std::string& msg = messages.front();
            try {
                std::cout << msg << std::endl;
            } catch (...) {}

            if (_logFile.is_open()) {
                _logFile << removeAnsiColors(msg) << std::endl;
            }
            messages.pop();
        }

        if (_logFile.is_open())
            _logFile.flush();
    }
}

std::string Logger::getTimestamp()
{
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S")
       << "." << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

std::string Logger::removeAnsiColors(const std::string& text)
{
    return std::regex_replace(text, std::regex("\033\\[[0-9;]*m"), "");
}

void Logger::log(const std::string& msg)
{
    std::string formattedMessage = getTimestamp() + " | " + msg;
    {
        std::lock_guard<std::mutex> lock(_queueMutex);
        _messageQueue.push(std::move(formattedMessage));
    }
    _condition.notify_one();
}

void Logger::debug(const std::string& msg)
{
    log("\033[36m[DEBUG]\033[0m " + msg);
}

void Logger::info(const std::string& msg)
{
    log("\033[32m[INFO]\033[0m " + msg);
}

void Logger::warning(const std::string& msg)
{
    log("\033[33m[WARNING]\033[0m " + msg);
}

void Logger::error(const std::string& msg)
{
    log("\033[31m[ERROR]\033[0m " + msg);
}