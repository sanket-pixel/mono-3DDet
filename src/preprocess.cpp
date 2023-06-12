#include "preprocess.hpp"

void MonoconPreProcessor::normalization(cv::Mat input_image){

    //  int original_width = input_image.size().width;
    // int original_height = input_image.size().height;
    input_image.convertTo(input_image, CV_32F);
    cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
    // Define the mean and standard deviation values
    cv::Scalar mean(123.675, 116.28, 103.53);
    cv::Scalar stdDev(58.395, 57.12, 57.375);

    // Subtract the mean from each channel
    cv::Mat subtracted_image;
    cv::subtract(input_image, mean, subtracted_image);

    // Divide the subtracted image by the standard deviation
    cv::Mat normalized_image;
    cv::divide(subtracted_image, stdDev, normalized_image);

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            // Access the element at (i, j)
            cv::Vec3b pixel = normalized_image.at<cv::Vec3b>(i, j);
            std::cout << "(" << static_cast<int>(pixel[0]) << ", "
                      << static_cast<int>(pixel[1]) << ", "
                      << static_cast<int>(pixel[2]) << ") ";
        }
        std::cout << std::endl;
    }
    // // Assign the normalized image to the output
    // output_image = normalized_image;
}