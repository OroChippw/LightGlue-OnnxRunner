/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description: 
*********************************/
#pragma once

#include <iostream>

#include "Configuration.h"


class BaseFeatureMatchOnnxRunner {
public:
    virtual int InitOrtEnv(Configuration cfg) 
    {
        return EXIT_SUCCESS;
    }

    virtual void SetMatchThresh(float thresh){}

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> InferenceImage(Configuration cfg , \
            const cv::Mat& srcImage, const cv::Mat& destImage)
    {
        return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>();
    };

    virtual std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> GetKeypointsResult()
    {
        return std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>();
    };
};