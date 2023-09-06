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
            InputNodeNames.emplace_back(_strdup(session->GetInputNameAllocated(i , allocator).get()));
            InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            // std::cout << "input_name.get() : " << input_name.get() << std::endl;
        }

        const size_t numOutputNodes = session->GetOutputCount();
        OutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            OutputNodeNames.emplace_back(_strdup(session->GetOutputNameAllocated(i , allocator).get()));
            OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
            // std::cout << "output_name.get() : " << output_name.get() << std::endl;
        }
        // std::cout << "[INFO] numInputNodes : " << numInputNodes << " numOutputNodes : " << numOutputNodes << std::endl;
        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

cv::Mat LightGlueOnnxRunner::PreProcess(Configuration cfg , const cv::Mat& Image)
{
	std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;
    cv::Mat resultImage = NormalizeImage(ResizeImage(Image ,cfg.image_size , scale));
    std::cout << "[INFO] Scale from 1.0f to "<< scale << std::endl;
    if (cfg.extractorType == "superpoint")
    {
        resultImage = RGB2Grayscale(resultImage);
    }

    return resultImage;
}

int LightGlueOnnxRunner::Inference(Configuration cfg , const cv::Mat& src , const cv::Mat& dest)
{   
    try 
    {   
        // Dynamic InputNodeShapes is [1,3,-1,-1]  
        // std::cout << "src : " << src.size() << std::endl;
        // std::cout << "dest : " << dest.size() << std::endl;

        InputNodeShapes[0][2] = src.size[0];
        InputNodeShapes[0][3] = src.size[1];
        InputNodeShapes[1][2] = dest.size[0];
        InputNodeShapes[1][3] = dest.size[1];

        std::vector<float> srcInputTensorValues(src.total());
        srcInputTensorValues.assign(src.begin<float>() , src.end<float>());

        std::vector<float> destInputTensorValues(dest.total());
        destInputTensorValues.assign(dest.begin<float>() , dest.end<float>());

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , srcInputTensorValues.data() , src.total() , \
            InputNodeShapes[0].data() , InputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , destInputTensorValues.data() , dest.total() , \
            InputNodeShapes[1].data() , InputNodeShapes[1].size()
        ));

        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = session->Run(Ort::RunOptions{nullptr} , InputNodeNames.data() , input_tensors.data() , \
                    input_tensors.size() , OutputNodeNames.data() , OutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = time_end - time_start;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        output_tensors = std::move(output_tensor);

        std::cout << "[INFO] LightGlueOnnxRunner inference finish ..." << std::endl;
	    std::cout << "[INFO] Inference cost time : " << diff.count() << "s" << std::endl;
    } 
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueOnnxRunner inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}


int LightGlueOnnxRunner::PostProcess(Configuration cfg)
{
    try{
        std::vector<int64_t> kpts0_Shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts0 = (int64_t*)output_tensors[0].GetTensorMutableData<void>();
        // 在Python里面是一个（batch = 1 , kpts_num , 2）的array，那么在C++里输出的长度就应该是kpts_num * 2
        // int kpts0_length = kpts0_Shape[1] * 2;
        printf("[RESULT INFO] kpts0 Shape : (%lld , %lld , %lld)\n" , kpts0_Shape[0] , kpts0_Shape[1] , kpts0_Shape[2]);

        std::vector<int64_t> kpts1_Shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts1 = (int64_t*)output_tensors[1].GetTensorMutableData<void>();
        // int kpts1_length = kpts1_Shape[1] * 2;
        printf("[RESULT INFO] kpts1 Shape : (%lld , %lld , %lld)\n" , kpts1_Shape[0] , kpts1_Shape[1] , kpts1_Shape[2]);

        std::vector<int64_t> matches0_Shape = output_tensors[2].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches0 = (int64_t*)output_tensors[2].GetTensorMutableData<void>();
        int match0_Counts = matches0_Shape[1];
        printf("[RESULT INFO] matches0 Shape : (%lld , %lld)\n" , matches0_Shape[0] , matches0_Shape[1]);

        std::vector<int64_t> matches1_Shape = output_tensors[3].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches1 = (int64_t*)output_tensors[3].GetTensorMutableData<void>();
        int match1_Counts = matches1_Shape[1];
        printf("[RESULT INFO] matches1 Shape : (%lld , %lld)\n" , matches1_Shape[0] , matches1_Shape[1]);

        std::vector<int64_t> mscore0_Shape = output_tensors[4].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores0 = (float*)output_tensors[4].GetTensorMutableData<void>();
        std::vector<int64_t> mscore1_Shape = output_tensors[5].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores1 = (float*)output_tensors[5].GetTensorMutableData<void>();

        // Process kpts0 and kpts1
        std::vector<cv::Point2f> kpts0_f , kpts1_f;
        for (int i = 0; i < kpts0_Shape[1] * 2; i += 2) 
        {
            kpts0_f.emplace_back(cv::Point2f(
                (kpts0[i] + 0.5) / scale - 0.5 , (kpts0[i + 1] + 0.5) / scale - 0.5));
        }
        for (int i = 0; i < kpts1_Shape[1] * 2; i += 2) 
        {
            kpts1_f.emplace_back(cv::Point2f(
                (kpts1[i] + 0.5) / scale - 0.5 , (kpts1[i + 1] + 0.5) / scale - 0.5)
            );
        }

        // Create match indices
        std::vector<int64_t> validIndices;
        for (int i = 0; i < matches0_Shape[1]; ++i) {
            if (matches0[i] > -1) { validIndices.emplace_back(i);}
        }

        std::vector<int64_t> matches;
        std::vector<cv::Point2f> m_kpts0 , m_kpts1;
        for (int i : validIndices) {
            matches.emplace_back(i);
            matches.emplace_back(matches0[i]);
        }
        for (size_t i = 0; i < matches.size(); i += 2) {
            if (i < matches.size() && matches[i] < matches0_Shape[1]) 
            {
                m_kpts0.emplace_back(kpts0_f[i]);
                m_kpts1.emplace_back(kpts1_f[i + 1]);
            }
        }

        keypoints_result.first = m_kpts0;
        keypoints_result.second = m_kpts1;
        
        std::cout << "[INFO] Postprocessing operation completed successfully" << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueOnnxRunner::InferenceImage(Configuration cfg , 
        const cv::Mat& srcImage, const cv::Mat& destImage)
{   
    std::cout << "< - * -------- INFERENCEIMAGE START -------- * ->" << std::endl;

    if (srcImage.empty() || destImage.empty())
	{
		throw  "[ERROR] ImageEmptyError ";
	}
    cv::Mat srcImage_copy = cv::Mat(srcImage);
    cv::Mat destImage_copy = cv::Mat(destImage);

    std::cout << "[INFO] => PreProcess srcImage" << std::endl;
    cv::Mat src = PreProcess(cfg , srcImage_copy);
    std::cout << "[INFO] => PreProcess destImage" << std::endl;
    cv::Mat dest = PreProcess(cfg , destImage_copy);

    Inference(cfg , src , dest);

    PostProcess(cfg);

    return GetKeypointsResult();
}

float LightGlueOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void LightGlueOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

LightGlueOnnxRunner::LightGlueOnnxRunner(unsigned int threads) : \
    num_threads(threads)
{
}

LightGlueOnnxRunner::~LightGlueOnnxRunner()
{
}
