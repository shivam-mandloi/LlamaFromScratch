#pragma once

#include "grid.hpp"
#include <string>
#include "HelpingFunction.hpp"

class tokenizer
{
public:
    tokenizer() 
    {
        std::string vocabStr = ReadTxtFile("vocabCPP.txt");
        std::string mergeStr = ReadTxtFile("mergesCPP.txt");

        
    }

    void encode()
    {
    }

    void decode()
    {
    }
};