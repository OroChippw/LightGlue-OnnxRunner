/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.09.03
    Description:
*********************************/
#pragma once

#include "LightGlueDecoupleOnnxRunner.h"

int LightGlueDecoupleOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LightGlueDecoupleOnnxRunner");
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
            const wchar_t* extractor_modelPath = multi_Byte_To_Wide_Char(cfg.extractorPath);
            const wchar_t* matcher_modelPath = multi_Byte_To_Wide_Char(cfg.lightgluePath);
        #else
            const char* extractor_modelPath = cfg.extractorPath;
            const char* matcher_modelPath = cfg.lightgluePath;
        #endif // _WIN32

        ExtractorSession = std::make_unique<Ort::Session>(env , extractor_modelPath , session_options);
        MatcherSession = std::make_unique<Ort::Session>(env , matcher_modelPath , session_options);

        // Initial Extractor 
        size_t numInputNodes = ExtractorSession->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            ExtractorInputNodeNames.emplace_back(_strdup(ExtractorSession->GetInputNameAllocated(i , allocator).get()));
            ExtractorInputNodeShapes.emplace_back(ExtractorSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        size_t numOutputNodes = ExtractorSession->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            ExtractorOutputNodeNames.emplace_back(_strdup(ExtractorSession->GetOutputNameAllocated(i , allocator).get()));            
            ExtractorOutputNodeShapes.emplace_back(ExtractorSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numInputNodes = 0;
        numOutputNodes = 0;
        
        // Initial Matcher 
        numInputNodes = MatcherSession->GetInputCount();
        ExtractorInputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            MatcherInputNodeNames.emplace_back(_strdup(MatcherSession->GetInputNameAllocated(i , allocator).get()));
            MatcherInputNodeShapes.emplace_back(MatcherSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        numOutputNodes = MatcherSession->GetOutputCount();
        ExtractorOutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            MatcherOutputNodeNames.emplace_back(_strdup(MatcherSession->GetOutputNameAllocated(i , allocator).get()));
            MatcherOutputNodeShapes.emplace_back(MatcherSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        delete extractor_modelPath;
        delete matcher_modelPath;

        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

cv::Mat LightGlueDecoupleOnnxRunner::Extractor_PreProcess(Configuration cfg , const cv::Mat& Image , float& scale)
{
	float temp_scale = scale;
    cv::Mat tempImage = Image.clone();
    std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;
    if (cfg.extractorType == "superpoint")
    {
        std::cout << "[INFO] ExtractorType Superpoint turn RGB to Grayscale" << std::endl;
        tempImage = RGB2Grayscale(tempImage);
    }
    std::string fn = "max";
    std::string interp = "area";
    cv::Mat resultImage = NormalizeImage(ResizeImage(tempImage , cfg.image_size , scale , fn , interp));
    std::cout << "[INFO] Scale from "<< temp_scale << " to "<< scale << std::endl;
   
    return resultImage;
}

int LightGlueDecoupleOnnxRunner::Extractor_Inference(Configuration cfg , const cv::Mat& image)
{   
    std::cout << "< - * -------- Extractor Inference START -------- * ->"<< std::endl;
    try 
    {   
        // Dynamic InputNodeShapes is [1,3,-1,-1] or [1,1,-1,-1]
        std::cout << "[INFO] Image Size : " << image.size() << " Channels : " << image.channels() << std::endl;
        
        // Build src input node shape and destImage input node shape
        int srcInputTensorSize , destInputTensorSize;
        if (cfg.extractorType == "superpoint")
        {
            ExtractorInputNodeShapes[0] = {1 , 1 , image.size().height , image.size().width};
        }else if (cfg.extractorType == "disk")
        {
            ExtractorInputNodeShapes[0] = {1 , 3 , image.size().height , image.size().width};
        }
        srcInputTensorSize = ExtractorInputNodeShapes[0][0] * ExtractorInputNodeShapes[0][1] \
                        * ExtractorInputNodeShapes[0][2] * ExtractorInputNodeShapes[0][3];

        std::vector<float> srcInputTensorValues(srcInputTensorSize);

        if (cfg.extractorType == "superpoint")
        {
            srcInputTensorValues.assign(image.begin<float>() , image.end<float>());
        }else{             
            int height = image.rows;
            int width = image.cols;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    cv::Vec3f pixel = image.at<cv::Vec3f>(y, x); // RGB
                    srcInputTensorValues[y * width + x] = pixel[2];
                    srcInputTensorValues[height * width + y * width + x] = pixel[1];
                    srcInputTensorValues[2 * height * width + y * width + x] = pixel[0];
                }
            }
        }
        
        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, \
                            OrtMemType::OrtMemTypeCPU);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , srcInputTensorValues.data() , srcInputTensorValues.size() , \
            ExtractorInputNodeShapes[0].data() , ExtractorInputNodeShapes[0].size()
        ));

        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = ExtractorSession->Run(Ort::RunOptions{nullptr} , ExtractorInputNodeNames.data() , input_tensors.data() , \
                    input_tensors.size() , ExtractorOutputNodeNames.data() , ExtractorOutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = time_end - time_start;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }

        extractor_outputtensors.emplace_back(std::move(output_tensor));

        std::cout << "[INFO] LightGlueDecoupleOnnxRunner Extractor inference finish ..." << std::endl;
	    std::cout << "[INFO] Extractor inference cost time : " << diff.count() << "s" << std::endl;
    } 
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Extractor inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}


std::pair<std::vector<cv::Point2f> , float*> LightGlueDecoupleOnnxRunner::Extractor_PostProcess(Configuration cfg , std::vector<Ort::Value> tensor)
{
    std::pair<std::vector<cv::Point2f> , float*> extractor_result;
    try{
        std::vector<int64_t> kpts_Shape = tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* kpts = (int64_t*)tensor[0].GetTensorMutableData<void>();
        for (int i = 0 ; i < kpts_Shape[1] ; i++)
        {
            std::cout << kpts[i] << " ";
        }
        printf("[RESULT INFO] kpts Shape : (%lld , %lld , %lld)\n" , kpts_Shape[0] , kpts_Shape[1] , kpts_Shape[2]);

        std::vector<int64_t> score_Shape = tensor[1].GetTensorTypeAndShapeInfo().GetShape();
        float* scores = (float*)tensor[1].GetTensorMutableData<void>();

        std::vector<int64_t> descriptors_Shape = tensor[2].GetTensorTypeAndShapeInfo().GetShape();
        float* desc = (float*)tensor[2].GetTensorMutableData<void>();
        printf("[RESULT INFO] desc Shape : (%lld , %lld , %lld)\n" , descriptors_Shape[0] , descriptors_Shape[1] , descriptors_Shape[2]);

        // Process kpts and descriptors
        std::vector<cv::Point2f> kpts_f;
        for (int i = 0; i < kpts_Shape[1] * 2; i += 2) 
        {
            kpts_f.emplace_back(cv::Point2f(kpts[i] , kpts[i + 1]));
        }

        extractor_result.first = kpts_f;
        extractor_result.second = desc;

        std::cout << "[INFO] Extractor postprocessing operation completed successfully" << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] Extractor postprocess failed : " << ex.what() << std::endl;
    }

    return extractor_result;
}

std::vector<cv::Point2f> LightGlueDecoupleOnnxRunner::Matcher_PreProcess(std::vector<cv::Point2f> kpts, int h , int w)
{
    return NormalizeKeypoints(kpts , h , w);
}


int LightGlueDecoupleOnnxRunner::Matcher_Inference(std::vector<cv::Point2f> kpts0 , \
            std::vector<cv::Point2f> kpts1 , float* desc0 , float* desc1)
{
    std::cout << "< - * -------- Matcher Inference START -------- * ->"<< std::endl;
    try
    {
        MatcherInputNodeShapes[0] = {1 , static_cast<int>(kpts0.size()) , 2};
        MatcherInputNodeShapes[1] = {1 , static_cast<int>(kpts1.size()) , 2};
        MatcherInputNodeShapes[2] = {1 , static_cast<int>(kpts0.size()) , 256};
        MatcherInputNodeShapes[3] = {1 , static_cast<int>(kpts1.size()) , 256};

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);

        float* kpts0_data = new float[kpts0.size() * 2];
        float* kpts1_data = new float[kpts1.size() * 2];

        for (size_t i = 0; i < kpts0.size(); ++i) {
            kpts0_data[i * 2] = kpts0[i].x;
            kpts0_data[i * 2 + 1] = kpts0[i].y;
        }
        for (size_t i = 0; i < kpts1.size(); ++i) {
            kpts1_data[i * 2] = kpts1[i].x;
            kpts1_data[i * 2 + 1] = kpts1[i].y;
        }

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts0_data , kpts0.size() * 2 * sizeof(float), \
            MatcherInputNodeShapes[0].data() , MatcherInputNodeShapes[0].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , kpts1_data , kpts1.size() * 2 * sizeof(float), \
            MatcherInputNodeShapes[1].data() , MatcherInputNodeShapes[1].size()
        ));
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc0 , kpts0.size() * 256 * sizeof(float), \
            MatcherInputNodeShapes[2].data() , MatcherInputNodeShapes[2].size()
        ));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , desc1 , kpts1.size() * 256 * sizeof(float) , \
            MatcherInputNodeShapes[3].data() , MatcherInputNodeShapes[3].size()
        ));


        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = MatcherSession->Run(Ort::RunOptions{nullptr} , MatcherInputNodeNames.data() , input_tensors.data() , \
                    input_tensors.size() , MatcherOutputNodeNames.data() , MatcherOutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = time_end - time_start;

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        matcher_outputtensors = std::move(output_tensor);

        std::cout << "[INFO] LightGlueDecoupleOnnxRunner Matcher inference finish ..." << std::endl;
        std::cout << "[INFO] Matcher inference cost time : " << diff.count() << "s" << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] LightGlueDecoupleOnnxRunner Matcher inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

int LightGlueDecoupleOnnxRunner::Matcher_PostProcess(Configuration cfg , std::vector<cv::Point2f> kpts0, std::vector<cv::Point2f> kpts1)
{
    try{
        std::vector<int64_t> matches0_Shape = matcher_outputtensors[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches0 = (int64_t*)matcher_outputtensors[0].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches0 Shape : (%lld , %lld)\n" , matches0_Shape[0] , matches0_Shape[1]);

        std::vector<int64_t> matches1_Shape = matcher_outputtensors[1].GetTensorTypeAndShapeInfo().GetShape();
        int64_t* matches1 = (int64_t*)matcher_outputtensors[1].GetTensorMutableData<void>();
        printf("[RESULT INFO] matches1 Shape : (%lld , %lld)\n" , matches1_Shape[0] , matches1_Shape[1]);

        std::vector<int64_t> mscore0_Shape = matcher_outputtensors[2].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores0 = (float*)matcher_outputtensors[2].GetTensorMutableData<void>();
        std::vector<int64_t> mscore1_Shape = matcher_outputtensors[3].GetTensorTypeAndShapeInfo().GetShape();
        float* mscores1 = (float*)matcher_outputtensors[3].GetTensorMutableData<void>();

        // Process kpts0 and kpts1
        std::vector<cv::Point2f> kpts0_f , kpts1_f;

        for (int i = 0; i < kpts0.size(); i++) 
        {
            kpts0_f.emplace_back(cv::Point2f(
                (kpts0[i].x + 0.5) / scales[0] - 0.5 , (kpts0[i].y + 0.5) / scales[0] - 0.5));
        }
        for (int i = 0; i < kpts1.size(); i++) 
        {
            kpts1_f.emplace_back(cv::Point2f(
                (kpts1[i].x + 0.5) / scales[0] - 0.5 , (kpts1[i].y + 0.5) / scales[0] - 0.5));
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

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::InferenceImage(Configuration cfg , 
        const cv::Mat& srcImage, const cv::Mat& destImage)
{   
    std::cout << "< - * -------- INFERENCEIMAGE START -------- * ->" << std::endl;

    if (srcImage.empty() || destImage.empty())
	{
		throw  "[ERROR] ImageEmptyError ";
	}
    cv::Mat srcImage_copy = cv::Mat(srcImage);
    cv::Mat destImage_copy = cv::Mat(destImage);

    // Extract Keypoints
    std::cout << "[INFO] => PreProcess srcImage" << std::endl;
    cv::Mat src = Extractor_PreProcess(cfg , srcImage_copy , scales[0]);
    std::cout << "[INFO] => PreProcess destImage" << std::endl;
    cv::Mat dest = Extractor_PreProcess(cfg , destImage_copy , scales[1]);
    
    Extractor_Inference(cfg , src);
    Extractor_Inference(cfg , dest);


    auto src_extract = Extractor_PostProcess(cfg , std::move(extractor_outputtensors[0]));
    auto dest_extract = Extractor_PostProcess(cfg , std::move(extractor_outputtensors[1]));

    // Build Matches
    auto normal_kpts0 = Matcher_PreProcess(src_extract.first , src.rows , src.cols);
    auto normal_kpts1 = Matcher_PreProcess(dest_extract.first , dest.rows , dest.cols);

    Matcher_Inference(normal_kpts0 , normal_kpts1 , src_extract.second , dest_extract.second);

    Matcher_PostProcess(cfg , src_extract.first , dest_extract.first);

    return GetKeypointsResult();
}

float LightGlueDecoupleOnnxRunner::GetMatchThresh()
{
    return this->matchThresh;
}

void LightGlueDecoupleOnnxRunner::SetMatchThresh(float thresh)
{
    this->matchThresh = thresh;
}

std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> LightGlueDecoupleOnnxRunner::GetKeypointsResult()
{
    return this->keypoints_result;
}

LightGlueDecoupleOnnxRunner::LightGlueDecoupleOnnxRunner(unsigned int threads) : \
    num_threads(threads)
{
}

LightGlueDecoupleOnnxRunner::~LightGlueDecoupleOnnxRunner()
{
}
