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

# --- OpenCV ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect dnn face)

# --- Qt ---
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)

# --- Dlib ---
find_package(dlib REQUIRED)

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

# --- Path to model files ---
set(MODELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/model)
target_compile_definitions(imgorg PRIVATE MODELS_PATH="${MODELS_DIR}")

# --- Debug mode definition ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(imgorg PRIVATE DEBUG_MODE)
endif()

# --- Link libraries ---
target_link_libraries(imgorg PRIVATE
    ${OpenCV_LIBS}
    Qt6::Widgets
    Qt6::Core
    Qt6::Gui
    dlib::dlib
)

# --- Include directories ---
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
    ${dlib_INCLUDE_DIRS}
)

# --- Compiler optimizations ---
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(imgorg PRIVATE -O3 -march=native)
endif()

# --- Install target ---
install(TARGETS imgorg DESTINATION bin)

# --- Notes for model files ---
message(STATUS "Remember to place the following model files in accessible paths:")
message(STATUS "  - shape_predictor_68_face_landmarks.dat  (Dlib)")
message(STATUS "  - lbfmodel.yaml                          (OpenCV LBF)")

message(STATUS "imgorg - Face Organizer Project Configured Successfully")
