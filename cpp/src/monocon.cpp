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

    std::cout << "Input Image " << mEngine->getBindingDimensions(0) << std::endl; 
    std::cout << "Calib  " << mEngine->getBindingDimensions(1) << std::endl; 
    std::cout << "Calib Inv " << mEngine->getBindingDimensions(2) << std::endl; 
    std::cout << "bboxes labels " << mEngine->getBindingDimensions(3) << std::endl; 
    std::cout << "bboxes 2d " << mEngine->getBindingDimensions(4) << std::endl; 
    std::cout << "bboxes 3d " << mEngine->getBindingDimensions(5) << std::endl; 
    return true;
}


cv::Mat Monocon::read_image(std::string image_path){
    return cv::imread(image_path,cv::IMREAD_COLOR);
}

bool Monocon::enqueue_input(float* host_buffer, cv::Mat image){
    nvinfer1::Dims input_dims = mEngine->getBindingDimensions(0);
    for (size_t batch = 0; batch < 1; ++batch) {
  
        int offset = input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * batch;
        int r = 0 , g = 0, b = 0;
        
        for (int i = 0; i < input_dims.d[1] * input_dims.d[2] * input_dims.d[3]; ++i) {
            if (i % 3 == 0) {
                host_buffer[offset + r++] = *(reinterpret_cast<float*>(image.data) + i);
            } else if (i % 3 == 1) {
                host_buffer[offset + g++ + input_dims.d[2] * input_dims.d[3]] = *(reinterpret_cast<float*>(image.data) + i);
            } else {
                host_buffer[offset + b++ + input_dims.d[2] * input_dims.d[3] * 2] = *(reinterpret_cast<float*>(image.data) + i);
            }
        }
    }    
}

std::vector<float*> Monocon::dequeue_boxes(float* output_flattened){
    const int num_boxes = mEngine->getBindingDimensions(5).d[1];
    const int dims_boxes = mEngine->getBindingDimensions(5).d[2];
    int counter = 0;
    std::vector<float*> boxes_vector;
    for (int i = 0; i < num_boxes; i++) {
        float* box = new float[dims_boxes];
        for (int j = 0; j < dims_boxes; j++){
            box[j] = output_flattened[counter];
            counter++;
        }
        boxes_vector.push_back(box);
    }
    return boxes_vector;
}

Eigen::Matrix<float, 3, 4> Monocon::read_calibration_file(std::string calib_path)
{
    std::ifstream file(calib_path);
    std::string line;
    Eigen::Matrix<float, 3, 4> calibrationMatrix;

    while (std::getline(file, line))
    {
        if (line.find("P_rect_02:") != std::string::npos)
        {
            std::istringstream iss(line);
            std::string keyword;
            float value;

            iss >> keyword;  // Read the keyword
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    iss >> value;  // Read the matrix values
                    calibrationMatrix(i, j) = value;
                }
            }

            break;  // Exit the loop after finding the calibration matrix
        }
    }

    return calibrationMatrix;
}

Eigen::Matrix4f Monocon::invertCalib(const Eigen::Matrix<float, 3, 4>& calib)
{
    Eigen::Matrix4f matrix;
    matrix.block<3, 4>(0, 0) = calib;
    matrix.row(3) << 0, 0, 0, 1;


    Eigen::Matrix4f invertedMatrix = matrix.inverse();
    return invertedMatrix;
}

bool Monocon::preprocess(cv::Mat img, cv::Mat &preprocessed_img ){
    std::cout << img.size() << std::endl;
    monocon_preprocessor.normalization(img, preprocessed_img);
    monocon_preprocessor.padding(preprocessed_img, preprocessed_img);
    std::cout << preprocessed_img.size() << std::endl;
    
}



void Monocon::get_bindings(){
   
    // Create the execution context
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    samplesCommon::BufferManager bufferManager{mEngine};

    // Read image from Disk 
    float* host_buffer_image = (float*)bufferManager.getHostBuffer("image");
    cv::Mat img = read_image(mParams.ioPathsParams.image_path);
    cv::Mat preprocessed_image;
    Monocon::preprocess(img, preprocessed_image);
    // Populate host buffer with input image.
    enqueue_input(host_buffer_image, preprocessed_image);

    // // Read calibration matrix
    float* host_buffer_calib = (float*)bufferManager.getHostBuffer("calib");
    Eigen::Matrix<float, 3, 4> calib = read_calibration_file(mParams.ioPathsParams.calib_path);
    size_t buffer_size_calib = bufferManager.size("calib");
    std::memcpy(host_buffer_calib, calib.data(), buffer_size_calib);

    // invert calib 
    float* host_buffer_inv_calib = (float*)bufferManager.getHostBuffer("calib_inv");
    Eigen::Matrix4f invertedMatrix = invertCalib(calib);
    size_t buffer_size_inv_calib = bufferManager.size("calib_inv");
    std::memcpy(host_buffer_inv_calib, invertedMatrix.data(), buffer_size_inv_calib);

     // Copy input from host to device
    bufferManager.copyInputToDevice();

     // Perform inference
    bool status_0 = context->executeV2(bufferManager.getDeviceBindings().data()); 

    //  Copy output to host
    bufferManager.copyOutputToHost(); 

     // convert boxes to vector
    float* output_flattened = static_cast<float*>(bufferManager.getHostBuffer("bboxes_3d"));
    std::vector<float*> boxes_vector = dequeue_boxes(output_flattened);
    std::cout << boxes_vector.at(0)[2] << std::endl;
   

}


