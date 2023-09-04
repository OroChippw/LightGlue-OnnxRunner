/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

void plotImages(const std::vector<cv::Mat>& Images , const std::vector<std::string>& Titles = std::vector<std::string>() , \
                const std::vector<std::string>& Cmaps = std::vector<std::string>() , int dpi = 100, double pad = 0.5, \
                bool adaptive = true)
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
    int n = static_cast<int>(Images.size());
    if (Titles.size() || Cmaps.size() != n)
    {
        throw std::runtime_error("[ERROR] The size of Titles、Cmaps and Images should be same");
    }

    std::vector<double> ratios(n);
    for (const auto& image : Images) {
        if (adaptive) {
            ratios.push_back(static_cast<double>(image.cols) / image.rows);
        } else {
            ratios.push_back(4.0 / 3.0);
        }
    }
    double totalRatio = 0.0;
    for (const double ratio : ratios) {
        totalRatio += ratio;
    }
    
    cv::Size figureSize(static_cast<int>(totalRatio * 4.5), 4.5);
    cv::Mat figure(figureSize, CV_8UC3, cv::Scalar(255, 255, 255));
    int x = 0;
    for (int i = 0; i < n; ++i) {
        const cv::Mat& img = Images[i];
        const std::string& title = Titles[i];
        const std::string& cmap = Cmaps[i];
        
        int width = static_cast<int>(img.rows * ratios[i]);
        cv::Mat subplot = figure(cv::Rect(x, 0, width, img.rows));
        
        cv::cvtColor(img, subplot, cv::COLOR_BGR2RGB);
        
        if (!title.empty()) {
            cv::putText(subplot, title, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0), 2);
        }
        
        x += width;
    }
    
    cv::imshow("Image Plot", figure);
    cv::waitKey(0);
    cv::destroyAllWindows();
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
    cv::Mat imgMatches;
    cv::drawMatches(img0, kpts0, img1, kpts1, matches, imgMatches, color);
    
    for (const auto& pt : kpts0) {
        cv::circle(imgMatches, pt, ps, color, -1);
    }

    for (const auto& pt : kpts1) {
        cv::circle(imgMatches, pt, ps, color, -1);
    }

    cv::imshow("Matches", imgMatches);
    cv::waitKey(0);
    cv::destroyAllWindows();
}