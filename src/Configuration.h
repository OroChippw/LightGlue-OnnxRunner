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
    std::string modelPath;
    std::string extractorType;
    bool isEndtoEnd = true;

    unsigned int image_size = 1024; 

    std::string device;
};