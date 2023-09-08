cmake_minimum_required(VERSION 3.0)
project(LightGlue-OnnxRunner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    # Set the MSVC input code encoding format to utf-8
    set(CMAKE_C_FLAGS /source-charset:utf-8)
endif()

message(STATUS "Project: LightGlue-OnnxRunner")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

include(cmake/platform.cmake)  # checking platform

# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
set(ONNXRUNTIME_DIR  ${CMAKE_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.14.1)
message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories("${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_LIB onnxruntime onnxruntime_providers_shared)

set(OPENCV_DIR  ${CMAKE_SOURCE_DIR}/third_party/opencv480/build)
message(STATUS "OPENCV_DIR Path: ${OPENCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OPENCV_DIR}/include)
link_directories("${OPENCV_DIR}/x64/vc16/lib")
set(OpenCV_LIB opencv_world480)


# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)
add_executable(main ${SRC_LIST})
target_compile_features(main PRIVATE cxx_std_17)

target_link_libraries(main ${OpenCV_LIB} ${ONNXRUNTIME_LIB} ${OpenCV_LIBS})
