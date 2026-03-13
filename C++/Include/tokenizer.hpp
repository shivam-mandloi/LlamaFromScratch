#pragma once

#include "grid.hpp"
#include <string>
#include "HelpingFunction.hpp"

class tokenizer
{
public:
    tokenizer() 
    {
        auto start = std::chrono::steady_clock::now();
        
        std::string vocabStr = ReadTxtFile("vocabCPP.txt");
        std::string mergeStr = ReadTxtFile("mergesCPP.txt");
        
        auto middle = std::chrono::steady_clock::now();

        std::vector<std::string> vocabArr = SplitString(vocabStr, " ");
        std::vector<std::string> mergeArr = SplitString(mergeStr, "<^>");

        auto end = std::chrono::steady_clock::now();

        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto fileReadingTime = std::chrono::duration_cast<std::chrono::milliseconds>(middle - start);
        auto splitTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - middle);

        std::cout << vocabArr.size() << " " << mergeArr.size() << std::endl;
        std::cout << "Total Time: " << totalTime.count() << "ms | File Reading Time: " << fileReadingTime.count() << "ms | Split Time: " << splitTime.count() << "ms" << std::endl; 
    }

    void encode()
    {

    }

    void decode()
    {
    }
};