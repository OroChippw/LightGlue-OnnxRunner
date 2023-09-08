/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description: 
*********************************/
#pragma once

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "viz2d.h"
#include "Configuration.h"
#include "LightGlueOnnxRunner.h"
#include "LightGlueDecoupleOnnxRunner.h"

std::vector<cv::Mat> ReadImage(std::vector<cv::String> image_filelist , bool grayscale = false)
{
    /*
    Func:
        Read an image from path as RGB or grayscale

    */
    int mode = cv::IMREAD_COLOR;
    if (grayscale)
    {
        mode = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    }

    std::vector<cv::Mat> image_matlist;
    for (const auto& file : image_filelist)
    {
        // std::cout << "[FILE INFO] : " << file << std::endl;
        cv::Mat image = cv::imread(file , mode);
        if (image.empty())
        {
            throw std::runtime_error("[ERROR] Could not read image at " + file);
        }
        if (!grayscale)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // BGR -> RGB
        }
        image_matlist.emplace_back(image);
    }

    return image_matlist;
}


int main(int argc , char* argv[])
{
    /* ****** CONFIG START ****** */
    std::string model_path = "${OnnxModelPath}";
    // Type of feature extractor. Supported extractors are 'superpoint' and 'disk'.
    std::string extractor_type = "${ModelExtractorType}";
    // Sample image size for ONNX tracing , resize the longer side of the images to this value. Supported image size {512 , 1024 , 2048}
    unsigned int image_size = 512; 
    bool grayscale = false;
    bool end2end = true;
    std::string device = "${Device}"; // Now only support "cpu"
    bool viz = true;
    float matchThresh = 0.0f;

    std::string image_path1 = "${YourImageDirPath1}";
    std::string image_path2 = "${YourImageDirPath2}";
    std::string save_path = "${YourResultSavePath}";

    /* ****** CONFIG END ****** */
    
    /* ****** Usage Example Start ****** */
    model_path = "D:\\OroChiLab\\LightGlue-OnnxRunner\\models\\superpoint\\superpoint_lightglue_end2end.onnx";
    extractor_type = "SuperPoint";

    // model_path = "D:\\OroChiLab\\LightGlue\\weights\\onnx\\disk_lightglue_end2end.onnx";
    // extractor_type = "Disk";

    image_path1 = "D:\\OroChiLab\\LightGlue\\data\\dir_0";
    image_path2 = "D:\\OroChiLab\\LightGlue\\data\\dir_1";
    device = "cpu";
    /* ****** Usage Example End ****** */

    Configuration cfg;
    cfg.modelPath = model_path;
    cfg.extractorType = extractor_type;
    cfg.isEndtoEnd = end2end;
    cfg.grayScale = grayscale;
    cfg.image_size = image_size;
    cfg.threshold = matchThresh;
    cfg.device = device;
    cfg.viz = viz;

    std::transform(cfg.extractorType.begin() , cfg.extractorType.end() , \
            cfg.extractorType.begin() , ::tolower);
    if (cfg.extractorType != "superpoint" && cfg.extractorType != "disk")
    {
        std::cerr << "[ERROR] Unsupported feature extractor type: " << extractor_type << std::endl;

        return EXIT_FAILURE;
    }else
    {
        std::cout << "[INFO] Extractor Type : " << cfg.extractorType << std::endl;
    }

    std::vector<cv::String> image_filelist1;
    std::vector<cv::String> image_filelist2;
    cv::glob(image_path1 , image_filelist1);
    cv::glob(image_path2 , image_filelist2);
    if (image_filelist1.size() != image_filelist2.size())
    {
        std::cout << "[INFO] Image Matlist1 size : " << image_filelist1.size() << std::endl;
        std::cout << "[INFO] Image Matlist2 size : " << image_filelist2.size() << std::endl;
        std::cerr << "[ERROR] The number of images in the source folder and \
                    the destination folder is inconsistent" << std::endl;

        return EXIT_FAILURE;
    }

    std::cout << "[INFO] => Building Image Matlist1" << std::endl;
    std::vector<cv::Mat> image_matlist1 = ReadImage(image_filelist1 , cfg.grayScale);
    std::cout << "[INFO] => Building Image Matlist2" << std::endl;
    std::vector<cv::Mat> image_matlist2 = ReadImage(image_filelist2 , cfg.grayScale);

    /* - * -------- End to End Example -------- * - */
    // Init Onnxruntime Env
    LightGlueOnnxRunner FeatureMatcher(std::thread::hardware_concurrency());
    FeatureMatcher.InitOrtEnv(cfg);
    FeatureMatcher.SetMatchThresh(cfg.threshold);
    
    auto iter1 = image_matlist1.begin();
    auto iter2 = image_matlist2.begin();

    for (;iter1 != image_matlist1.end() && iter2 !=image_matlist2.end() ; \
            ++iter1, ++iter2)
    {
        auto startTime = std::chrono::steady_clock::now();
        auto kpts_result = FeatureMatcher.InferenceImage(cfg , *iter1 , *iter2);
        auto endTime = std::chrono::steady_clock::now();

        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        std::cout << "[INFO] LightGlueOnnxRunner single picture whole process takes time : " \
                    << elapsedTime << " ms" << std::endl;
        if (cfg.viz)
        {
            std::vector<cv::Mat> imagesPair = {*iter1 , *iter2};
            std::vector<std::string> titlePair = {"srcImage" , "destImage"};
            cv::Mat figure = plotImages(imagesPair , kpts_result , titlePair);
        }
        auto kpts = FeatureMatcher.GetKeypointsResult();
    }
    
    return EXIT_SUCCESS;
}


