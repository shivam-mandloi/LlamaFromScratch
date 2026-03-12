#pragma once

#include <string>
#include <fstream>


std::string weightPathLocation = "/mnt/c/Users/shiva/Desktop/IISC/LLAMA/LLAMA_Weights";

std::string ReadTxtFile(std::string fileName)
{
    std::string path = weightPathLocation + "/" + fileName;
    std::fstream newFile;
    std::string temp;
    newFile.open(path, std::ios::in);
    if (!newFile.is_open())
    {
        std::cerr << "Error: Could not open file " << path << std::endl;
        exit(0);
    }
    std::string res = "";
    while (getline(newFile, temp))
    {
        if (temp != "")
            res += temp;
    }
    return res;
}