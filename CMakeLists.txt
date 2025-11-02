cmake_minimum_required(VERSION 3.18)
project(imgorg VERSION 1.0.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE Release)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


find_program(NVCC nvcc PATHS /opt/cuda/bin /usr/local/cuda/bin)
if(NVCC)
    message(STATUS "Found nvcc: ${NVCC}")
    set(CMAKE_CUDA_COMPILER ${NVCC})
    enable_language(CUDA)
    set(HAVE_CUDA TRUE)
else()
    message(STATUS "nvcc not found - building CPU only")
    set(HAVE_CUDA FALSE)
endif()


find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui objdetect dnn)


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

set(ALL_SOURCES
    ${MAIN_SOURCE}
    ${CORE_SOURCES}
    ${FACE_SOURCES}
    ${FILESYSTEM_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
)

add_executable(imgorg ${ALL_SOURCES})
target_link_libraries(imgorg ${OpenCV_LIBS})

if(HAVE_CUDA)
    target_link_libraries(imgorg cudart)
    target_compile_definitions(imgorg PRIVATE USE_CUDA_ACCELERATION)
endif()


target_include_directories(imgorg PRIVATE  
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FaceRecognition
    ${CMAKE_CURRENT_SOURCE_DIR}/src/FileSystem
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GPU   
    ${CMAKE_CURRENT_SOURCE_DIR}/Utils
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_features(imgorg PRIVATE cxx_std_17)
target_compile_options(imgorg PRIVATE -O3 -march=native)

install(TARGETS imgorg DESTINATION bin)

message(STATUS "imgorg - CUDA: ${HAVE_CUDA}")