#pragma once
#include <string>
#include <fstream>
#include <mutex>
#include <thread>
#include <queue>
#include <filesystem>
#include <condition_variable>

class Logger
{
public:
    static Logger& instance(const std::string& filename = "imgorg.log");

    void log(const std::string& msg);
    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warning(const std::string& msg);
    void error(const std::string& msg);

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    Logger(Logger&&) = delete;
    Logger& operator=(Logger&&) = delete;

private:
    explicit Logger(const std::string& filename);
    ~Logger();

    void workerThread();
    std::string getTimestamp();
    std::string removeAnsiColors(const std::string& text);

    std::ofstream _logFile;
    std::queue<std::string> _messageQueue;
    std::mutex _queueMutex;
    std::condition_variable _condition;
    std::thread _workerThread;
    bool _stopRequested = false;
};