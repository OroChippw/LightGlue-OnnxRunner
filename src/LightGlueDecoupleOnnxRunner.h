/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.09.07
    Description: 
*********************************/
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "Configuration.h"


class LightGlueDecoupleOnnxRunner
{
private:
	const unsigned int num_threads;

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    float matchThresh = 0.0f;
    
private:
    cv::Mat PreProcess(Configuration cfg , const cv::Mat& srcImage , float& scale);
    int Inference(Configuration cfg , const cv::Mat& src , const cv::Mat& dest);
    int PostProcess(Configuration cfg);

public:
    explicit LightGlueDecoupleOnnxRunner(unsigned int num_threads = 1);
    ~LightGlueDecoupleOnnxRunner();

    float GetMatchThresh();
    void SetMatchThresh(float thresh);
};