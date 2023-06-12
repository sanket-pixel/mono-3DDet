/*!
 @file monocon.cpp
 @author Sanket Rajendra Shah (sanket.shah@motor-ai.com)
 @brief 
 @version 0.1
 @date 2023-05-11
 
 @copyright Copyright (c) 2023
 
 */
#include "monocon.hpp"
// #include "postprocess.hpp"
// #include "preprocess.hpp"



// #include "include/preprocess.h"


//!
//! \brief Uses a ONNX parser to create the Onnx Monocon Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx Monocon network
//!
//! \param builder Pointer to the engine builder
//!

bool Monocon::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mParams.engineParams.OnnxFilePath.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        sample::gLogError<< "Onnx model cannot be parsed ! " << std::endl;
        return false;
    }
    builder->setMaxBatchSize(BATCH_SIZE_);
    // config->setMaxWorkspaceSize(2_GiB); //8_GiB);

    if (mParams.engineParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.engineParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    if (mParams.engineParams.dlaCore >=0 ){
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.engineParams.dlaCore);
    sample::gLogInfo << "Deep Learning Acclerator (DLA) was enabled . \n";
    }
    return true;
}



//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx Monocon network by parsing the Onnx model and builds
//!          the engine that will be used to run Monocon (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
std::shared_ptr<nvinfer1::ICudaEngine> Monocon::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return nullptr;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return nullptr;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return nullptr;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return nullptr;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return nullptr;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return nullptr;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return nullptr;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return nullptr;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        sample::gLogError << "Failed to create engine \n";
        return nullptr;
    }
    
    std::ofstream engineFile(mParams.engineParams.SerializedEnginePath, std::ios::binary);
    engineFile.write(static_cast<const char*>(plan->data()), plan->size());
    engineFile.close();
    // ASSERT(network->getNbInputs() == 1);
    // mInputDims = network->getInput(0)->getDimensions();
    // ASSERT(mInputDims.nbDims == 4);

    // ASSERT(network->getNbOutputs() == 4);
    // mOutputDims = network->getOutput(4)->getDimensions();
    // ASSERT(mOutputDims.nbDims == 2);

    return mEngine;
}

bool Monocon::buildFromSerializedEngine(){

    // Load serialized engine from file
    std::ifstream engineFileStream(mParams.engineParams.SerializedEnginePath, std::ios::binary);
    engineFileStream.seekg(0, engineFileStream.end);
    const size_t engineSize = engineFileStream.tellg();
    engineFileStream.seekg(0, engineFileStream.beg);
    std::unique_ptr<char[]> engineData(new char[engineSize]);
    engineFileStream.read(engineData.get(), engineSize);
    engineFileStream.close();
    // Create the TensorRT runtime
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));   
     // Deserialize the TensorRT engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(engineData.get(), engineSize));   

    // std::cout << "Input Image " << mEngine->getBindingDimensions(0) << std::endl; 
    // std::cout << "Calib  " << mEngine->getBindingDimensions(1) << std::endl; 
    // std::cout << "Calib Inv " << mEngine->getBindingDimensions(2) << std::endl; 
    // std::cout << "bboxes labels " << mEngine->getBindingDimensions(3) << std::endl; 
    // std::cout << "bboxes 2d " << mEngine->getBindingDimensions(4) << std::endl; 
    // std::cout << "bboxes 3d " << mEngine->getBindingDimensions(5) << std::endl; 
    return true;
}


cv::Mat Monocon::read_image(std::string image_path){
    return cv::imread(image_path,cv::IMREAD_COLOR);
}

bool Monocon::preprocess(cv::Mat img, cv::Mat &preprocessed_img ){
    MonoconPreProcessor monocon_preprocess(mParams.modelParams.resized_image_size_width,
                                           mParams.modelParams.resized_image_size_height);
    monocon_preprocess.normalization(img, preprocessed_img);
    
    // monocon_preprocess.initialize_globals(input_dims.d[3],input_dims.d[2]);
    // monocon_preprocess.basic_preprocessing(img, preprocessed_img);
    // monocon_preprocess.norm_scaling(preprocessed_img, cv::Scalar(0.485f, 0.456f, 0.406f),
    //                                         cv::Scalar(0.229f, 0.224f, 0.225f));

}



void Monocon::get_bindings(){
   
    // Create the execution context
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());

    // // Populate host buffer with input image.
    // samplesCommon::BufferManager bufferManager{mEngine};
    // float* host_buffer = (float*)bufferManager.getHostBuffer("input");


    // // Read image from Disk 
    cv::Mat img = read_image(mParams.ioPathsParams.image_path);
    std::cout << img.size() << std::endl;
    cv::Mat preprocessed_image;
    Monocon::preprocess(img, preprocessed_image);
    std::cout << preprocessed_image.at<cv::Vec3f>(0,0)[0] <<std::endl;
    // Monocon::enqueue_input(host_buffer, preprocessed_image);

   

}


