# LightGlue-OnnxRunner
## Introduction
LightGlue-OnnxRunner is an example using 

## Development Enviroments
>  - Windows 11 Professional 
>  - CUDA v11.7
>  - cmake version 3.26.2

## Quick Start

### Requirements
``` 
# onnxruntime 3rdparty
This repository use onnxruntime-win-x64-1.14.1
# opencv 3rdparty
This repository use opencv4.7.0
# CXX_STANDARD 17
```
### Build and Run
```
# Enter the source code directory where CMakeLists.txt is located, and create a new build folder
mkdir build
# Enter the build folder and run CMake to configure the project
cd build
cmake ..
# Use the build system to compile/link this project
cmake --build .
# If the specified compilation mode is debug or release, it is as follows
# cmake --build . --config Debug
# cmkae --build . --config Release
```
### Model Checkpoints(TODO)



### License
This project is licensed under the MIT License.