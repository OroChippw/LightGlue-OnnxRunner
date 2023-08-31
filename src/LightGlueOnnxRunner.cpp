/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include "utils.h"
#include "transform.h"
#include "LightGlueOnnxRunner.h"

int LightGlueOnnxRunner::InitOrtEnv(Configuration cfg)
{
    try
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueOnnxRunner");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda") {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};
            cuda_options.device_id = 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }

        #if _WIN32
            std::cout << "[INFO] Env _WIN32 change modelpath from multi byte to wide char ..." << std::endl;
            const wchar_t* modelPath = multi_Byte_To_Wide_Char(cfg.modelPath);
        #else
            const char* modelPath = cfg.modelPath;
        #endif // _WIN32

        session = std::make_unique<Ort::Session>(env , modelPath , session_options);

        const size_t numInputNodes = session->GetInputCount();
        InputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            auto input_name = session->GetInputNameAllocated(i , allocator);
            InputNodeNames.emplace_back(strdup(input_name.get()));
            InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            // std::cout << "input_name.get() : " << input_name.get() << std::endl;
        }

        const size_t numOutputNodes = session->GetOutputCount();
        OutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            auto output_name = session->GetOutputNameAllocated(i , allocator);
            OutputNodeNames.emplace_back(strdup(output_name.get()));
            OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            // std::cout << "output_name.get() : " << output_name.get() << std::endl;
        }
        std::cout << "[INFO] numInputNodes : " << numInputNodes << " numOutputNodes : " << numOutputNodes << std::endl;
        std::cout << "[INFO] ONNX Runtime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNX Runtime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

cv::Mat LightGlueOnnxRunner::PreProcess(Configuration cfg , const cv::Mat& srcImage)
{
	std::cout << "[INFO] srcImage info :  width : " << srcImage.cols << " height :  " << srcImage.rows << std::endl;
    cv::Mat resultImage = NormalizeImage(ResizeImageByLongestSide(srcImage , cfg.image_size));

    return resultImage;
}

int LightGlueOnnxRunner::Inference(Configuration cfg , const cv::Mat& src , const cv::Mat& dest)
{

    return EXIT_SUCCESS;
}

int LightGlueOnnxRunner::PostProcess()
{
    return EXIT_SUCCESS;
}

cv::Mat LightGlueOnnxRunner::InferenceImage(Configuration cfg , 
        const cv::Mat& srcImage, const cv::Mat& destImage)
{   
    if (srcImage.empty() || destImage.empty())
	{
		throw  "[ERROR] Image EmptyError ";
	}
    cv::Mat src = PreProcess(cfg , srcImage);
    cv::Mat dest = PreProcess(cfg , destImage);

    Inference(cfg , src , dest);
    PostProcess();

    return cv::Mat();
}

LightGlueOnnxRunner::LightGlueOnnxRunner(unsigned int threads) : \
    num_threads(threads)
{

}

LightGlueOnnxRunner::~LightGlueOnnxRunner()
{
}
