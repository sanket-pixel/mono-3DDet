#include "preprocess.hpp"

void MonoconPreProcessor::normalization(cv::Mat input_image, cv::Mat &output_image){

    cv::Mat bgrImage;
    cv::cvtColor(input_image, bgrImage, cv::COLOR_RGB2BGR);

    cv::Mat float_image;
    bgrImage.convertTo(float_image, CV_32FC3);

    // Define the mean and standard deviation values
    cv::Scalar mean(123.675, 116.28, 103.53);
    cv::Scalar stdDev(58.395, 57.12, 57.375);

   
    // Subtract the mean from each channel
    cv::Mat subtracted_image;
    cv::subtract(float_image, mean, subtracted_image);

     // Divide the subtracted image by the standard deviation
    cv::Mat normalized_image;
    cv::divide(subtracted_image, stdDev, normalized_image);
   
    output_image = normalized_image;

}


void MonoconPreProcessor::padding(cv::Mat input_image, cv::Mat &output_image){

    cv::Mat bgrImage;
    cv::cvtColor(input_image, bgrImage, cv::COLOR_RGB2BGR);

    cv::Mat float_image;
    bgrImage.convertTo(float_image, CV_32FC3);

    // Define the mean and standard deviation values
    cv::Scalar mean(123.675, 116.28, 103.53);
    cv::Scalar stdDev(58.395, 57.12, 57.375);

   
    // Subtract the mean from each channel
    cv::Mat subtracted_image;
    cv::subtract(float_image, mean, subtracted_image);

     // Divide the subtracted image by the standard deviation
    cv::Mat normalized_image;
    cv::divide(subtracted_image, stdDev, normalized_image);
   
    output_image = normalized_image;

}