/*********************************
    Copyright: OroChippw
    Author: OroChippw
    Date: 2023.08.31
    Description:
*********************************/
#pragma once

#include <iostream>

struct Configuration
{
    std::string lightgluePath;
    std::string extractorPath;
    
    std::string extractorType;
    bool isEndtoEnd = true;
    bool grayScale = false;

    unsigned int image_size = 512; 
    float threshold = 0.0f;

    std::string device;
    bool viz = false;
};