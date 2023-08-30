# SegmentAnything-OnnxRunner
## Introduction
SegmentAnything-OnnxRunner is an example using Meta AI Research's SAM onnx model.The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image.This repository is used to record the experiment data that run SAM onnx model on CPU/GPU.At the same time, the encoder and decoder of SAM are decoupled in this repository.

Currently, the interface only supports GPU execution.The specific experimental data and equipment used are shown below. And the inferface is only supported on Windows and may encounter issues when running on Linux.

## Development Enviroments
>  - Windows 10 Professional 
>  - CUDA v11.3
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
The SAM encoder and decoder are decoupled and quantized. After decoupling, if you perform multiple interactive clicks on a picture, you don't need to re-encode it. The model with -quantize is the quantized version

### Experiment Record
Environment Device : i5-13600KF + NVIDIA GeForce RTX 3060（12GB）
Input image resolution : 1920 * 1080 * 3  
All models are available in [Baidu Pan](https://pan.baidu.com/s/1j0z1mHDOshOCcQWwetmFnQ?pwd=ljgr) (code: ljgr).    
#### Encoder
| Encoder version | Model Size(MB/GB) | CPU encoding speed(s) | 
| :------------------:| :---------------: | :---------------: | 
| sam_vit_b_01ec64_encoder.onnx          | 342MB | 2.5485 | 
| sam_vit_b_01ec64_encoder-quantize.onnx | 103MB | 2.0446 | 
| sam_vit_l_0b3195_encoder.onnx          | 1.14GB | 6.0346 | 
| sam_vit_l_0b3195_encoder-quantize.onnx | 316MB | 4.1599 | 

#### Decoder
| Decoder version | Model Size(MB) | CPU decoding speed(s) | 
| :------------------:| :---------------: | :---------------: | 
| sam_vit_b_01ec64_decoder.onnx            | 15.7MB | 0.075 | 
| sam_vit_b_01ec64_decoder_singlemask.onnx | 15.7MB | 0.075 | 
| sam_vit_b_01ec64_decoder.onnx            | 15.7MB | 0.086 | 
| sam_vit_b_01ec64_decoder_singlemask.onnx | 15.7MB | 0.082 | 


### License
This project is licensed under the MIT License.