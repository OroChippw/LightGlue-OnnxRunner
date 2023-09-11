/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.09.07
    Description: 
*********************************/
#pragma once

#ifndef LIGHTGLUE_DECOUPLE_ONNX_RUNNER_H
#define LIGHTGLUE_DECOUPLE_ONNX_RUNNER_H

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "utils.h"
#include "transform.h"
#include "BaseOnnxRunner.h"
#include "Configuration.h"


class LightGlueDecoupleOnnxRunner : public BaseFeatureMatchOnnxRunner
{
private:
	const unsigned int num_threads;

    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> ExtractorSession , MatcherSession;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char*> ExtractorInputNodeNames;
    std::vector<std::vector<int64_t>> ExtractorInputNodeShapes;
    std::vector<char*> ExtractorOutputNodeNames;
    std::vector<std::vector<int64_t>> ExtractorOutputNodeShapes;

    std::vector<char*> MatcherInputNodeNames;
    std::vector<std::vector<int64_t>> MatcherInputNodeShapes;
    std::vector<char*> MatcherOutputNodeNames;
    std::vector<std::vector<int64_t>> MatcherOutputNodeShapes;

    float matchThresh = 0.0f;

    std::vector<float> scales = {1.0f , 1.0f};

    std::vector<std::vector<Ort::Value>> extractor_outputtensors; // 因为要存src和dest的两个结果，所以用嵌套vector
    std::vector<Ort::Value> matcher_outputtensors;

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> keypoints_result;
    
private:
    cv::Mat Extractor_PreProcess(Configuration cfg , const cv::Mat& srcImage , float& scale);
    int Extractor_Inference(Configuration cfg , const cv::Mat& image);
    std::pair<std::vector<cv::Point2f> , float*> Extractor_PostProcess(Configuration cfg , std::vector<Ort::Value> tensor);

    std::vector<cv::Point2f> Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h , int w);
    int Matcher_Inference(Configuration cfg , std::vector<cv::Point2f> kpts0 , std::vector<cv::Point2f> kpts1 , float* desc0 , float* desc1);
    int Matcher_PostProcess(Configuration cfg , std::vector<cv::Point2f> kpts0 , std::vector<cv::Point2f> kpts1);

public:
    explicit LightGlueDecoupleOnnxRunner(unsigned int num_threads = 1);
    ~LightGlueDecoupleOnnxRunner();

    float GetMatchThresh();
    void SetMatchThresh(float thresh);

    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult();

    int InitOrtEnv(Configuration cfg);
    
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg , \
            const cv::Mat& srcImage, const cv::Mat& destImage);
};

#endif // LIGHT_GLUE_DECOUPLE_ONNX_RUNNER_H