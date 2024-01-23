/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include "LightGlueOnnxRunner.h"

int LightGlueOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
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
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0; 
            cuda_options.arena_extend_strategy = 1; // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options.AppendExecutionProvider_CUDA(cuda_options);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        }

        #if _WIN32
            std::cout << "[INFO] Env _WIN32 change modelpath from multi byte to wide char ..." << std::endl;
            const wchar_t* modelPath = multi_Byte_To_Wide_Char(cfg.lightgluePath);
        #else
            const char* modelPath = cfg.lightgluePath;
        #endif // _WIN32

        session = std::make_unique<Ort::Session>(env , modelPath , session_options);

        const size_t numInputNodes = session->GetInputCount();
        InputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            InputNodeNames.emplace_back(_strdup(session->GetInputNameAllocated(i , allocator).get()));
            InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        const size_t numOutputNodes = session->GetOutputCount();
        OutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            OutputNodeNames.emplace_back(_strdup(session->GetOutputNameAllocated(i , allocator).get()));
            OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        
        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

cv::Mat LightGlueOnnxRunner::PreProcess(Configuration cfg , const cv::Mat& Image , float& scale)
{
	float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;
    
    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resultImage = NormalizeImage(ResizeImage(tempImage ,cfg.image_size , scale , fn , interp));
    if (cfg.extractorType == "superpoint")
    {
        std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
        resultImage = RGB2Grayscale(resultImage);
    }
    std::cout << "[INFO] Scale from "<< temp_scale << " to "<< scale << std::endl;
   
    return resultImage;
}

int LightGlueOnnxRunner::Inference(Configuration cfg , const cv::Mat& src , const cv::Mat& dest)
{   
    try 
    {   
        // Dynamic InputNodeShapes is [1,3,-1,-1]  
        std::cout << "[INFO] srcImage Size : " << src.size() << " Channels : " << src.channels() << std::endl;
        std::cout << "[INFO] destImage Size : " << dest.size() << " Channels : " << src.channels() << std::endl;
        
        // Build src input node shape and destImage input node shape
        int srcInputTensorSize , destInputTensorSize;
        if (cfg.extractorType == "superpoint")
        {
            InputNodeShapes[0] = {1 , 1 , src.size().height , src.size().width};
            InputNodeShapes[1] = {1 , 1 , dest.size().height , dest.size().width};
        }else if (cfg.extractorType == "disk")
        {
            InputNodeShapes[0] = {1 , 3 , src.size().height , src.size().width};
            InputNodeShapes[1] = {1 , 3 , dest.size().height , dest.size().width};
        }
        srcInputTensorSize = InputNodeShapes[0][0] * InputNodeShapes[0][1] * InputNodeShapes[0][2] * InputNodeShapes[0][3];
        destInputTensorSize = InputNodeShapes[1][0] * InputNodeShapes[1][1] * InputNodeShapes[1][2] * InputNodeShapes[1][3];

        std::vector<float> srcInputTensorValues(srcInputTensorSize);
        std::vector<float> destInputTensorValues(destInputTensorSize);

        if (cfg.extractorType == "superpoint")
        {
            srcInputTensorValues.assign(src.begin<float>() , src.end<float>());
            destInputTensorValues.assign(dest.begin<float>() , dest.end<float>());
        }else{             
            int src_height = src.rows;
            int src_width = src.cols;
            for (int y = 0; y < src_height; y++) {
                for (int x = 0; x < src_width; x++) {
                    cv::Vec3f pixel = src.at<cv::Vec3f>(y, x); // RGB
                    srcInputTensorValues[y * src_width + x] = pixel[2];
                    srcInputTensorValues[src_height * src_width + y * src_width + x] = pixel[1];
                    srcInputTensorValues[2 * src_height * src_width + y * src_width + x] = pixel[0];
                }
            }
            int dest_height = dest.rows;
            int dest_width = dest.cols;
            for (int y = 0; y < dest_height; y++) {
                for (int x = 0; x < dest_width; x++) {
                    cv::Vec3f pixel = dest.at<cv::Vec3f>(y, x);
                    destInputTensorValues[y * dest_width + x] = pixel[2];
                    destInputTensorValues[dest_height * dest_width + y * dest_width + x] = pixel[1];
                    destInputTensorValues[2 * dest_height * dest_width + y * dest_width + x] = pixel[0];
                }
            }
        }

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , srcInputTensorValues.data() , srcInputTensorValues.size() , \
            InputNodeShapes[0].data() , InputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , destInputTensorValues.data() , destInputTensorValues.size() , \
            InputNodeShapes[1].data() , InputNodeShapes[1].size()
        ));

        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = session->Run(Ort::RunOptions{nullptr} , InputNodeNames.data() , input_tensors.data() , \
                    input_tensors.size() , OutputNodeNames.data() , OutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
        timer += diff;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        output_tensors = std::move(output_tensor);

        std::cout << "[INFO] LightGlueOnnxRunner inference finish ..." << std::endl;
	    std::cout << "[INFO] Inference cost time : " << diff << "ms" << std::endl;
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
        printf("[RESULT INFO] kpts0 Shape : (%lld , %lld , %lld)\n" , kpts0_Shape[0] , kpts0_Shape[1] , kpts0_Shape[2]);

        std::vector<int64_t> kpts1_Shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts1 = (int64_t*)output_tensors[1].GetTensorMutableData<void>();
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
                (kpts0[i] + 0.5) / scales[0] - 0.5 , (kpts0[i + 1] + 0.5) / scales[0] - 0.5));
        }
        for (int i = 0; i < kpts1_Shape[1] * 2; i += 2) 
        {
            kpts1_f.emplace_back(cv::Point2f(
                (kpts1[i] + 0.5) / scales[1] - 0.5 , (kpts1[i + 1] + 0.5) / scales[1] - 0.5)
            );
        }

        // Create match indices
        std::vector<int64_t> validIndices;
        for (int i = 0; i < matches0_Shape[1]; ++i) {
            if (matches0[i] > -1 && mscores0[i] > this->matchThresh && matches1[matches0[i]] == i) { 
                validIndices.emplace_back(i);
            }
        }

        std::set<std::pair<int, int> > matches;
        std::vector<cv::Point2f> m_kpts0 , m_kpts1;
        for (int i : validIndices) {
            matches.insert(std::make_pair(i, matches0[i]));
        }

        std::cout << "[RESULT INFO] matches Size : " << matches.size() << std::endl;

        for (const auto& match : matches) {
            m_kpts0.emplace_back(kpts0_f[match.first]);
            m_kpts1.emplace_back(kpts1_f[match.second]);
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
    cv::Mat src = PreProcess(cfg , srcImage_copy , scales[0]);
    std::cout << "[INFO] => PreProcess destImage" << std::endl;
    cv::Mat dest = PreProcess(cfg , destImage_copy , scales[1]);
    
    Inference(cfg , src , dest);

    PostProcess(cfg);

    output_tensors.clear();

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

double LightGlueOnnxRunner::GetTimer(std::string name="matcher")
{
    return this->timer;
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
