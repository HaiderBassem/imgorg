cmake_minimum_required(VERSION 3.18)
project(imgorg VERSION 1.0.0 LANGUAGES CXX)

# --- C++ standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)




# --- Build type (Debug / Release) ---
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Output folder ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Policy fix (for CUDA warning) ---
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 NEW)
endif()

# --- CUDA detection (optional) ---
find_program(NVCC nvcc PATHS /opt/cuda/bin /usr/local/cuda/bin)
if(NVCC)
    message(STATUS "Found nvcc: ${NVCC}")
    enable_language(CUDA)
    set(HAVE_CUDA TRUE)
else()
    message(STATUS "nvcc not found - building CPU only")
    set(HAVE_CUDA FALSE)
endif()

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect dnn)

# --- Qt ---
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)

# --- Source files ---
set(MAIN_SOURCE src/main.cpp)
set(CORE_SOURCES src/Core/FaceOrganizer.cpp)
set(FACE_SOURCES
    src/FaceRecognition/FaceDetector.cpp
    src/FaceRecognition/FaceMatcher.cpp
    src/FaceRecognition/FaceEmbedding.cpp
)
set(FILESYSTEM_SOURCES
    src/FileSystem/FolderScanner.cpp
    src/FileSystem/FileOrganizer.cpp
    src/FileSystem/PathUtils.cpp
)
set(UTILS_SOURCES
    src/Utils/Logger.cpp
    src/Utils/ProgressTracker.cpp
    src/Utils/ImageUtils.cpp
)
set(GPU_SOURCES src/GPU/CUDAAccelerator.cpp)
set(UI_SOURCES
    src/UI/mainwindow.h
    src/UI/mainwindow.cpp
    src/UI/mainwindow.ui
)


set(ALL_SOURCES
    ${MAIN_SOURCE}
    ${CORE_SOURCES}
    ${FACE_SOURCES}
    ${FILESYSTEM_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
    ${UI_SOURCES}
)

# --- Target ---
add_executable(imgorg ${ALL_SOURCES})

# --- Debug mode definition ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(imgorg PRIVATE DEBUG_MODE)
endif()

target_link_libraries(imgorg PRIVATE
    ${OpenCV_LIBS}
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
)

if(HAVE_CUDA)
    target_link_libraries(imgorg PRIVATE cudart)
    target_compile_definitions(imgorg PRIVATE USE_CUDA_ACCELERATION)
endif()

target_include_directories(imgorg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FaceRecognition
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FileSystem
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GPU
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils
    ${CMAKE_CURRENT_SOURCE_DIR}/src/UI
    ${OpenCV_INCLUDE_DIRS}
)

# --- Compiler optimizations ---
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(imgorg PRIVATE -O3 -march=native)
endif()

# --- Install target ---
install(TARGETS imgorg DESTINATION bin)

message(STATUS "imgorg - CUDA: ${HAVE_CUDA}")
