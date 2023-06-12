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
#include "preprocess.hpp"



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
    nvinfer1::Dims32 input_dims = mEngine->getBindingDimensions(0);
    MonoconPreProcessor monocon_preprocess(mParams.modelParams.resized_image_size_width,
                                           mParams.modelParams.resized_image_size_height);
    monocon_preprocess.normalization(img);
    // monocon_preprocess.initialize_globals(input_dims.d[3],input_dims.d[2]);
    // monocon_preprocess.basic_preprocessing(img, preprocessed_img);
    // monocon_preprocess.norm_scaling(preprocessed_img, cv::Scalar(0.485f, 0.456f, 0.406f),
    //                                         cv::Scalar(0.229f, 0.224f, 0.225f));

}

// bool Monocon::enqueue_input(float* host_buffer, cv::Mat preprocessed_img){
//     nvinfer1::Dims input_dims = mEngine->getBindingDimensions(0);
//     for (size_t batch = 0; batch < 1; ++batch) {
  
//         int offset = input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * batch;
//         int r = 0 , g = 0, b = 0;
//         for (int i = 0; i < input_dims.d[1] * input_dims.d[2] * input_dims.d[3]; ++i) {
//             if (i % 3 == 0) {
//                 host_buffer[offset + r++] = *(reinterpret_cast<float*>(preprocessed_img.data) + i);
//             } else if (i % 3 == 1) {
//                 host_buffer[offset + g++ + input_dims.d[2] * input_dims.d[3]] = *(reinterpret_cast<float*>(preprocessed_img.data) + i);
//             } else {
//                 host_buffer[offset + b++ + input_dims.d[2] * input_dims.d[3] * 2] = *(reinterpret_cast<float*>(preprocessed_img.data) + i);
//             }
//         }
//     }    
// }

// std::vector<float*> Monocon::dequeue_boxes(float* output_flattened){
//     const int num_boxes = mEngine->getBindingDimensions(4).d[0];
//     const int dims_boxes = mEngine->getBindingDimensions(4).d[1];
//     int counter = 0;
//     std::vector<float*> boxes_vector;
//     for (int i = 0; i < num_boxes; i++) {
//         float* box = new float[12];
//         for (int j = 0; j < dims_boxes; j++){
//             box[j] = output_flattened[counter];
//             counter++;
//         }
//         boxes_vector.push_back(box);
//     }
//     return boxes_vector;
// }

// bool Monocon::measure_latency(){

//     // Create the execution context
//     auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
        
//     // Populate host buffer with input image.
//     samplesCommon::BufferManager bufferManager{mEngine};
//     float* host_buffer = (float*)bufferManager.getHostBuffer("input");

//     // Get input Dimensions


//     // Read image from Disk 
//     cv::Mat img = read_image(mParams.ioPathsParams.image_path);
//     cv::Mat preprocessed_image;
    
//     // start measuring preprocessing time ( for CPU )
//     auto start_time = std::chrono::high_resolution_clock::now();
//     Monocon::preprocess(img, preprocessed_image);
//     Monocon::enqueue_input(host_buffer, preprocessed_image);
//     auto end_time = std::chrono::high_resolution_clock::now();
//     auto preprocessing_latency = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     std::cout << "Pre-Processing time : " << preprocessing_latency << " ms" << std::endl;
    
//     // measure time needed to copy image to GPU
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
    
//     // Copy input from host to device
//     cudaEventRecord(start);
//     bufferManager.copyInputToDevice();
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float host_to_device_latency;
//     cudaEventElapsedTime(&host_to_device_latency, start, stop);
//     std::cout << "Data transfer time from Host to Device : " << host_to_device_latency << " ms" << std::endl;
    
//     // Perform inference
//     cudaEventRecord(start);
//     bool status_0 = context->executeV2(bufferManager.getDeviceBindings().data());    
//     auto inference_chrono = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float inference_latency;
//     cudaEventElapsedTime(&inference_latency, start, stop);
//     std::cout << "Inference Latency : " << inference_latency << " ms" << std::endl;
    
//     // Copy output to host
//     cudaEventRecord(start);
//     bufferManager.copyOutputToHost();
//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     float device_to_host_latency;
//     cudaEventElapsedTime(&device_to_host_latency, start, stop);
//     std::cout << "Device to Host Latency : " << device_to_host_latency << " ms" << std::endl;
    
//     // convert enqueued output to vectors of arrays.
//     start_time = std::chrono::high_resolution_clock::now();
//     float* output_flattened = static_cast<float*>(bufferManager.getHostBuffer("2065"));
//     std::vector<float*> boxes_vector = dequeue_boxes(output_flattened);
    
//     // copy output to vector
//     std::vector<cv::Rect2d> final_boxes;
//     camera_processors::postprocessors::MonoconPostProcessor monocon_postprocess;
//     monocon_postprocess.post_processing(boxes_vector, mParams.modelParams.resized_image_size,
//                                          mParams.modelParams.original_width,  mParams.modelParams.original_height,
//                                          mParams.modelParams.conf_thresh,  mParams.modelParams.nms_thresh,  mParams.modelParams.num_classes,
//                                         final_boxes);
//     end_time = std::chrono::high_resolution_clock::now();
//     auto post_processing_latency = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//     std::cout << "Post-Processing time : " << post_processing_latency << " ms" << std::endl;
    
//     // Total latency
//     float total_latency = preprocessing_latency + host_to_device_latency + inference_latency + device_to_host_latency  + post_processing_latency;
//     std::cout << "Total Latency : " << total_latency << std::endl;
//     float fps = 1.0/(total_latency/1000.0);
//     std::cout << "Frame Rate Prediction : " << fps << " FPS" << std::endl;

   
//     return true;

// }

void Monocon::get_bindings(){
   
    // Create the execution context
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    

    // // Populate host buffer with input image.
    // samplesCommon::BufferManager bufferManager{mEngine};
    // float* host_buffer = (float*)bufferManager.getHostBuffer("input");


    // Read image from Disk 
    cv::Mat img = read_image(mParams.ioPathsParams.image_path);
    cv::Mat preprocessed_image;
    Monocon::preprocess(img, preprocessed_image);
    // Monocon::enqueue_input(host_buffer, preprocessed_image);

    // // Copy input from host to device
    // bufferManager.copyInputToDevice();
    
    // // Perform inference
    // bool status_0 = context->executeV2(bufferManager.getDeviceBindings().data());    

    // // Copy output to host
    // bufferManager.copyOutputToHost();

    // // convert enqueued output to vectors of arrays.
    // float* output_flattened = static_cast<float*>(bufferManager.getHostBuffer("2065"));
    // std::vector<float*> boxes_vector = dequeue_boxes(output_flattened);
    
    // std::vector<std::vector<float>> bindings;
    // // Iterate over bindings
    // for (int i = 0; i < mEngine->getNbBindings(); i++){
    //     context->getBindingDimensions(i);
    //     std::string bindingName = mEngine->getBindingName(i);
    //     float* binding_ptr = (float*)bufferManager.getHostBuffer(bindingName);
    //     int buffer_size = bufferManager.size(bindingName)/4;
    //     std::vector<float> buffer_data{binding_ptr, binding_ptr + buffer_size};
    //     bindings.emplace_back(buffer_data);
    // }

    // // copy output to vector
    // std::vector<cv::Rect2d> final_boxes;
    // camera_processors::postprocessors::MonoconPostProcessor monocon_postprocess;
    // monocon_postprocess.post_processing(boxes_vector, mParams.modelParams.resized_image_size,
    //                                      mParams.modelParams.original_width,  mParams.modelParams.original_height,
    //                                      mParams.modelParams.conf_thresh,  mParams.modelParams.nms_thresh,  mParams.modelParams.num_classes,
    //                                     final_boxes);

    // return bindings;

}


// std::vector<std::vector<float>>  get_cpp_bindings(){
//     Params params;
//     Monocon monocon(params); 
//     monocon.buildFromSerializedEngine();
//     return monocon.get_bindings();

// }