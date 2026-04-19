#pragma once

#include <string>
#include <fstream>
#include <vector>
#include <thread>

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

std::vector<std::string> SplitString(const std::string &str, const std::string &spliter)
{
    std::vector<std::string> res;

    if (str.empty() || spliter.empty())
        return res;

    size_t start = 0;
    size_t end = str.find(spliter);

    while (end != std::string::npos)
    {
        if (end > start)
        {
            res.emplace_back(str.substr(start, end - start));
        }
        start = end + spliter.length();
        end = str.find(spliter, start);
    }

    if (start < str.length())
    {
        res.emplace_back(str.substr(start));
    }

    return res;
}

void LoadBin(std::string filename, float *arr, size_t numElement)
{
    filename = weightPathLocation + "/" + filename;

    std::ifstream file(filename, std::ios::binary);

    if (!file)
        throw std::runtime_error("Could not open file");

    file.read(reinterpret_cast<char *>(arr), numElement * sizeof(float));
}

at::Tensor LoadTensor(std::string filename, std::vector<int64_t> shape)
{
    filename = weightPathLocation + "/" + filename;

    int64_t size = 1;
    for (int64_t ele : shape)
        size *= ele;
    at::Tensor tensor = at::empty(shape, at::kHalf);

    std::ifstream file(filename, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open file: " + filename);
    file.read(reinterpret_cast<char *>(tensor.data_ptr<at::Half>()),
              size * sizeof(at::Half));
    return tensor;
}