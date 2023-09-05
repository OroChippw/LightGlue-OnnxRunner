/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

void plotImages(const std::vector<cv::Mat>& Images , \
                std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> kpts_pair , \
                const std::vector<std::string>& Titles = std::vector<std::string>(1 , "image") , \
                const std::vector<std::string>& Cmaps = std::vector<std::string>(1 , "gray") , \
                int dpi = 100, bool adaptive = true)
{
    /*
    Func:
        Plot a set of images horizontally.
    Args:
        imgs: a list of cv::Mat images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    */
    try
    {
        int n = static_cast<int>(Images.size());
        std::vector<double> ratios;

        for (const auto& image : Images) {
            if (adaptive) {
                ratios.emplace_back(static_cast<double>(image.cols) / image.rows); // W / H
            } else {
                ratios.emplace_back(4.0 / 3.0);
            }
        }

        // 整个图像集的绘图尺寸。它的宽度是所有图像宽高比之和乘以4.5，高度固定为4.5
        double totalRatio = std::accumulate(ratios.begin() , ratios.end() , 0.0);
        double figureWidth = totalRatio * 4.5;
        cv::Size2f figureSize(static_cast<double>(figureWidth) * dpi, 4.5 * dpi);
        cv::Mat figure(figureSize, CV_8UC3, cv::Scalar(255, 255, 255));

        int x_offset = 0;
        for (unsigned int i = 0; i < n; ++i) {
            const cv::Mat& image = Images[i];
            cv::cvtColor(image , image , cv::COLOR_BGR2RGB);
            const std::string& title = Titles[i];
            // const std::string& cmap = Cmaps[i];
            
            cv::Mat resized_image;
            cv::Rect roi(cv::Point(x_offset, 0), cv::Size(static_cast<int>(ratios[i] * figureSize.height), figureSize.height));
            cv::resize(image , resized_image , roi.size());
            resized_image.copyTo(figure(roi));

            if (!title.empty()) {
                cv::putText(figure , title , cv::Point(x_offset + 10 , 30) , 
                            cv::FONT_HERSHEY_SIMPLEX , 1 , cv::Scalar(255,255,255) , 2);            
            }
            
            x_offset += resized_image.cols;
        }

        auto kpts0 = kpts_pair.first;
        auto kpts1 = kpts_pair.second;
        // std::cout << "kpts0.size() : " << kpts0.size() << std::endl;
        // for (int i = 0 ; i < kpts0.size() ; i++)
        // {
        //     // printf("%.2f " , kpts0[i]);
        //     std::cout << kpts0[i] << " ";
        // }
        // std::cout << "\nkpts1.size() : " << kpts1.size() << std::endl;
        // for (int i = 0 ; i < kpts1.size() ; i++)
        // {
        //     // printf("%.2f " , kpts1[i]);
        //     std::cout << kpts1[i] << " ";

        // }
        // // plotMatches();

        // cv::imshow("Image Plot", figure);
        // cv::waitKey(0);
        // cv::destroyAllWindows();
    }
    catch(const std::exception& e)
    {
        std::cerr << "[ERROR] PlotImagesError : " << e.what() << '\n';
    }
}

void plotKeypoints()
{
    /*
    Func:
        Plot matches for a pair of existing images.
    Args:
        kpts0, kpts1: corresponding keypoints of size (N, 2).
        color: color of each match, string or RGB tuple. Random if not given.
        lw: width of the lines.
        ps: size of the end points (no endpoint if ps=0)
        indices: indices of the images to draw the matches on.
        a: alpha opacity of the match lines.
    */

}

void plotMatches(const std::vector<cv::Point2f>& kpts0, const std::vector<cv::Point2f>& kpts1, \
            const cv::Scalar& color = cv::Scalar(0, 255, 0), float lw = 1.5, int ps = 4)
{
    assert(kpts0.size() == kpts1.size());
}