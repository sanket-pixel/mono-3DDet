#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#pragma once

class MonoconPreProcessor{
    private:
        int _resized_width;
        int _resized_height;

    public:
        MonoconPreProcessor(const int _resized_width, const int _resized_height){
            this->_resized_height = _resized_height;
            this->_resized_width = _resized_width; 

        }
        // void initialize_globals(int resized_width, int resized_height);
        void normalization(cv::Mat input_image, cv::Mat &output_image);
        void padding(cv::Mat input_image, cv::Mat &output_image);
        // void norm_scaling(cv::Mat &normalised_image, cv::Scalar norm_sub, cv::Scalar norm_div);
};
