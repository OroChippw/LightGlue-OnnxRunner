/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// #include <cuda_provider_factory.h>  // 若在GPU环境下运行可以使用cuda进行加速

#include "Configuration.h"

class LightGlueOnnxRunner
{
private:
	const unsigned int num_threads;

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> InputNodeNames;
    std::vector<std::vector<int64_t>> InputNodeShapes;

    std::vector<std::string> OutputNodeNames;
    std::vector<std::vector<int64_t>> OutputNodeShapes;

private:
    cv::Mat PreProcess(Configuration cfg , const cv::Mat& srcImage);
    int Inference(Configuration cfg , const cv::Mat& src , const cv::Mat& dest);
    int PostProcess();

public:
    explicit LightGlueOnnxRunner(unsigned int num_threads = 1);
    ~LightGlueOnnxRunner();

    int InitOrtEnv(Configuration cfg);
    
    cv::Mat InferenceImage(Configuration cfg , \
            const cv::Mat& srcImage, const cv::Mat& destImage);

};