/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include <opencv2/opencv.hpp>

cv::Mat NormalizeImage(cv::Mat& Image)
{
    cv::Mat normalizedImage;
    Image.convertTo(normalizedImage , CV_32F, 1.0 / 255.0);

    return normalizedImage;
}

void NormalizeKeypoints(cv::Mat& Image , int h , int w)
{
    
}

std::tuple<int , int> GetPreProcessShape(int old_h , int old_w , int long_side_length)
{
	double scale = long_side_length * 1.0 / MAX(old_h , old_w);
	int new_h = (int)(old_h * scale + 0.5);
	int new_w = (int)(old_w * scale + 0.5);
	std::tuple<int, int> newShape(new_h, new_w);
	return newShape;
}

cv::Mat ResizeImageByLongestSide(cv::Mat Image , int size , \
        const std::string& interp = "area")
{
    int mode = cv::INTER_AREA;
    if (interp == "linear") {
        mode = cv::INTER_LINEAR;
    } else if (interp == "cubic") {
        mode = cv::INTER_CUBIC;
    } else if (interp == "nearest") {
        mode = cv::INTER_NEAREST;
    }

    cv::Mat resizeImage;
	const unsigned int h = Image.rows;
	const unsigned int w = Image.cols;
	std::tuple<int, int> newShape = GetPreProcessShape(h , w , size);

	cv::resize(Image , resizeImage , \
        cv::Size(std::get<1>(newShape) , std::get<0>(newShape)) , mode);

	return resizeImage;
}

cv::Mat ResizeImage(const cv::Mat& Image, int size, float& scale , const std::string& fn="max", \
            const std::string& interp="area") {
    // Resize an image to a fixed size, or according to max or min edge.
    int h = Image.rows;
    int w = Image.cols;

    std::function<int(int, int)> func;
    if (fn == "max") {
        func = [](int a, int b) { return std::max(a, b); };
;
    }
    else if (fn == "min") {
        func = [](int a, int b) { return std::min(a, b); };
    }
    else {
        throw std::invalid_argument("[ERROR] Incorrect function: " + fn);
    }

    int h_new, w_new;
    if (size == 512 || size == 1024 || size == 2048) {
        scale = static_cast<float>(size) / static_cast<float>(func(h, w));
        h_new = static_cast<int>(round(h * scale));
        w_new = static_cast<int>(round(w * scale));
    }
    else {
        throw std::invalid_argument("Incorrect new size: " + std::to_string(size));
    }

    int mode;
    if (interp == "linear") {
        mode = cv::INTER_LINEAR;
    }
    else if (interp == "cubic") {
        mode = cv::INTER_CUBIC;
    }
    else if (interp == "nearest") {
        mode = cv::INTER_NEAREST;
    }
    else if (interp == "area") {
        mode = cv::INTER_AREA;
    }
    else {
        throw std::invalid_argument("[ERROR] Incorrect interpolation mode: " + interp);
    }

    cv::Mat resizeImage;
    cv::resize(Image, resizeImage, cv::Size(w_new, h_new), 0, 0, mode);

    return resizeImage;
}

cv::Mat RGB2Grayscale(cv::Mat& Image) {
    cv::Mat resultImage;
    cv::cvtColor(Image, resultImage, cv::COLOR_BGR2GRAY);

    return resultImage;
}
