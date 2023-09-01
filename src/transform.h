/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include <opencv2/opencv.hpp>

cv::Mat NormalizeImage(const cv::Mat& Image)
{
    cv::Mat normalizedImage;
    Image.convertTo(normalizedImage , CV_32F);
    normalizedImage /= 255.0;

    return normalizedImage;
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

cv::Mat RGB2Grayscale(cv::Mat& Image) {
    cv::Mat resultImage;
    cv::cvtColor(Image, resultImage, cv::COLOR_BGR2GRAY);

    return resultImage;
}

