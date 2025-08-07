#pragma once

#include <iostream>
#include <fstream>
#include <type_traits>
#include <utility>
#include <filesystem>
#include <mutex>
#include <string>

#include <algorithm>
#include <cctype>

inline std::mutex logMutex;

template<typename T, typename = void>
struct is_streamable : std::false_type {};

template<typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>> : std::true_type {};

template<typename T>
T check(const T& value) {
    if constexpr (is_streamable<T>::value) {
        std::cout << value << std::endl;
    } else {
        std::cout << "[Unprintable Type]" << std::endl;
    }
    return value;
}

static const std::filesystem::path getProjectRoot = [] {
    std::filesystem::path headerPath(__FILE__);
    std::filesystem::path projectRoot = headerPath.parent_path().parent_path().parent_path().parent_path();
    return projectRoot;
}();

static const std::filesystem::path getBaseLogDirectory = [] {
    return getProjectRoot / "log";
}();

template<typename T>
void logFile(const std::filesystem::path& path, const T& value) {
    const std::lock_guard<std::mutex> lock(logMutex);
    try {
        std::filesystem::path logDirectory = path.parent_path();

        if (!logDirectory.empty() && !std::filesystem::exists(logDirectory)) {
            std::filesystem::create_directories(logDirectory);
        }

        std::ofstream logFileStream(path, std::ios::app);

        if (!logFileStream.is_open()) {
            std::cerr << "Error: Could not open log file at " << path << std::endl;
            return;
        }

        if constexpr (is_streamable<T>::value) {
            logFileStream << value << std::endl;
        } else {
            logFileStream << "[Unable to Log Type]" << std::endl;
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error in logger: " << e.what() << std::endl;
    }
}

template<typename T>
void logToFile(const T& value) {
    static const std::filesystem::path defaultLogFilePath = getBaseLogDirectory / "log.txt";
    logFile(defaultLogFilePath, value);
}

template<typename T>
void logToFile(const char* subPath, const T& value) {
    std::filesystem::path fullLogFilePath = getBaseLogDirectory / subPath;

    if (!fullLogFilePath.has_extension()) {
        fullLogFilePath += ".txt";
    }
    logFile(fullLogFilePath, value);
}

// void clearLogDirectory() {
//     const std::lock_guard<std::mutex> lock(logMutex);
//     try {
//         std::filesystem::path logDir = getBaseLogDirectory;

//         if (std::filesystem::exists(logDir) && std::filesystem::is_directory(logDir)) {
//             std::cout << "Are you sure you want to delete all log files in: " << logDir.string() << "? (y/N): ";
//             std::string confirmation;
//             std::cin >> confirmation;

//             std::transform(confirmation.begin(), confirmation.end(), confirmation.begin(), [](unsigned char c) -> char { return static_cast<char>(std::tolower(c)); });


//             std::transform(confirmation.begin(), confirmation.end(), confirmation.begin(),[](unsigned char c){ return std::tolower(c); });

//             if (confirmation == "y") {
//                 std::cout << "Log directory cleared successfully." << std::endl;
//             } else {
//                 std::cout << "Log directory cleanup cancelled." << std::endl;
//             }
//         } else {
//             std::cout << "Log directory does not exist or is not a directory: " 
//                       << logDir.string() << std::endl;
//         }
//     } catch (const std::filesystem::filesystem_error& e) {
//         std::cerr << "Filesystem error during log directory cleanup: " << e.what() << std::endl;
//     }
// }