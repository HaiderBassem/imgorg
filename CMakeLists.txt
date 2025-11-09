cmake_minimum_required(VERSION 3.18)
project(imgorg VERSION 2.0.0 LANGUAGES CXX)

# ============================================================================
# C++ STANDARD
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# BUILD TYPE
# ============================================================================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================================
# FIND PACKAGES
# ============================================================================

# OpenCV (Required)
find_package(OpenCV REQUIRED 
    COMPONENTS 
        core 
        imgproc 
        highgui 
        objdetect 
        dnn 
        face
        features2d
        xfeatures2d
)

if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Dlib (Required)
find_package(dlib REQUIRED)

if(dlib_FOUND)
    message(STATUS "Dlib found")
else()
    message(FATAL_ERROR "Dlib not found!")
endif()

# Qt6 (Optional - for GUI)
find_package(Qt6 QUIET COMPONENTS Widgets Core Gui)

if(Qt6_FOUND)
    message(STATUS "Qt6 found - GUI will be built")
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    set(BUILD_GUI ON)
else()
    message(STATUS "Qt6 not found - Building CLI only")
    set(BUILD_GUI OFF)
endif()

# ============================================================================
# SOURCE FILES
# ============================================================================

# Main source
set(MAIN_SOURCE src/main.cpp)

# Core application
set(CORE_SOURCES
    src/Core/Application.cpp
)

# Face recognition
set(FACE_SOURCES
    src/FaceRecognition/FaceDetector.cpp
    src/FaceRecognition/FaceComparator.cpp
    src/FaceRecognition/FaceMatcher.cpp
    src/FaceRecognition/FaceEmbedding.cpp
)

# File system
set(FILESYSTEM_SOURCES
    src/FileSystem/FolderScanner.cpp
    src/FileSystem/FileOrganizer.cpp
    src/FileSystem/PathUtils.cpp
)

# Utilities
set(UTILS_SOURCES
    src/Utils/Logger.cpp
    src/Utils/ProgressTracker.cpp
    src/Utils/ImageUtils.cpp
)

# GPU (optional)
set(GPU_SOURCES
    src/GPU/CUDAAccelerator.cpp
)

# UI sources (if Qt is available)
if(BUILD_GUI)
    set(UI_SOURCES
        src/UI/mainwindow.h
        src/UI/mainwindow.cpp
        src/UI/mainwindow.ui
    )
endif()

# Combine all sources
set(ALL_SOURCES
    ${MAIN_SOURCE}
    ${CORE_SOURCES}
    ${FACE_SOURCES}
    ${FILESYSTEM_SOURCES}
    ${UTILS_SOURCES}
)

# Add GPU sources if available
if(CUDA_FOUND)
    list(APPEND ALL_SOURCES ${GPU_SOURCES})
endif()

# Add UI sources if building GUI
if(BUILD_GUI)
    list(APPEND ALL_SOURCES ${UI_SOURCES})
endif()

# ============================================================================
# EXECUTABLE TARGET
# ============================================================================
add_executable(imgorg ${ALL_SOURCES})

# ============================================================================
# INCLUDE DIRECTORIES
# ============================================================================
target_include_directories(imgorg PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FaceRecognition
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FileSystem
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GPU
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Utils
    ${OpenCV_INCLUDE_DIRS}
)

if(BUILD_GUI)
    target_include_directories(imgorg PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/UI
    )
endif()

# ============================================================================
# LINK LIBRARIES
# ============================================================================
target_link_libraries(imgorg PRIVATE
    ${OpenCV_LIBS}
    dlib::dlib
)

if(BUILD_GUI)
    target_link_libraries(imgorg PRIVATE
        Qt6::Widgets
        Qt6::Core
        Qt6::Gui
    )
endif()

# ============================================================================
# COMPILER DEFINITIONS
# ============================================================================

# Model paths
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model)
target_compile_definitions(imgorg PRIVATE 
    MODELS_PATH="${MODELS_DIR}"
)

# Debug mode
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(imgorg PRIVATE DEBUG_MODE)
    message(STATUS "Debug mode enabled")
endif()

# GUI flag
if(BUILD_GUI)
    target_compile_definitions(imgorg PRIVATE HAS_GUI)
endif()

# ============================================================================
# COMPILER FLAGS
# ============================================================================

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Optimization flags for release
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(imgorg PRIVATE
            -O3                  # Maximum optimization
            -march=native        # Optimize for current CPU
            -mtune=native
            -flto                # Link-time optimization
            -ffast-math          # Fast math operations
            -funroll-loops       # Loop unrolling
        )
        
        target_link_options(imgorg PRIVATE
            -flto                # Link-time optimization
        )
    elseif(MSVC)
        target_compile_options(imgorg PRIVATE
            /O2                  # Maximum optimization
            /GL                  # Whole program optimization
            /arch:AVX2          # Use AVX2 instructions
        )
        
        target_link_options(imgorg PRIVATE
            /LTCG               # Link-time code generation
        )
    endif()
    
    message(STATUS "Release optimizations enabled")
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug flags
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(imgorg PRIVATE
            -g                   # Debug symbols
            -O0                  # No optimization
            -Wall                # All warnings
            -Wextra              # Extra warnings
            -Wpedantic           # Pedantic warnings
        )
    elseif(MSVC)
        target_compile_options(imgorg PRIVATE
            /Zi                  # Debug info
            /Od                  # No optimization
            /W4                  # Warning level 4
        )
    endif()
endif()

# ============================================================================
# THREADING
# ============================================================================
find_package(Threads REQUIRED)
target_link_libraries(imgorg PRIVATE Threads::Threads)

# ============================================================================
# INSTALL
# ============================================================================
install(TARGETS imgorg 
    RUNTIME DESTINATION bin
)

# Install model files (if they exist)
if(EXISTS ${MODELS_DIR})
    install(DIRECTORY ${MODELS_DIR} 
        DESTINATION share/imgorg
    )
endif()

# ============================================================================
# INFORMATION
# ============================================================================
message(STATUS "")
message(STATUS "══════════════════════════════════════════════════════")
message(STATUS "  Face Organizer - Configuration Summary")
message(STATUS "══════════════════════════════════════════════════════")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:         ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  OpenCV version:   ${OpenCV_VERSION}")
message(STATUS "  Qt6 GUI:          ${BUILD_GUI}")
message(STATUS "  Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "══════════════════════════════════════════════════════")
message(STATUS "")

# ============================================================================
# REQUIRED MODEL FILES
# ============================================================================
message(STATUS "Required model files (place in 'model/' directory):")
message(STATUS "  - shape_predictor_68_face_landmarks.dat")
message(STATUS "  - lbfmodel.yaml")
message(STATUS "  - face_detection_yunet_2023mar.onnx")
message(STATUS "  - deploy.prototxt")
message(STATUS "  - res10_300x300_ssd_iter_140000.caffemodel")
message(STATUS "  - haarcascade_frontalface_default.xml")
message(STATUS "")