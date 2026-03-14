#pragma once

#include <string>
#include <unordered_map>

#include "grid.hpp"
#include "HelpingFunction.hpp"

class tokenizer
{
    std::string spaceEnc = "Ġ", newLine = "Ċ";
    std::unordered_map<std::string, int> mergeMap, vocabMap;
    std::vector<std::string> vocabArr;

public:
    tokenizer()
    {
        // Read vocab and merge file
        std::string vocabStr = ReadTxtFile("vocabCPP.txt");
        std::string mergeStr = ReadTxtFile("mergesCPP.txt");

        // Make vocab and merge vector
        vocabArr = SplitString(vocabStr, " ");
        std::vector<std::string> mergeArr = SplitString(mergeStr, "<^>");

        // Create map for faster access to elements
        for (int i = 0; i < mergeArr.size(); i++)
            mergeMap[mergeArr[i]] = i;
        for (int i = 0; i < vocabArr.size(); i++)
            vocabMap[vocabArr[i]] = i;
    }

    grid<int> encode(std::string text)
    {
        std::vector<std::string> splitedText;

        // split text by char and replace space and newLine by the Llama token
        for (int i = 0; i < text.size(); i++)
        {
            if (text[i] == ' ')
                splitedText.push_back(spaceEnc);
            else if (text[i] == '\n')
                splitedText.push_back(newLine);
            else
                splitedText.emplace_back(1, text[i]);
        }

        // loop until there is not any char left to merge
        while (true)
        {
            int mergeIndex = mergeMap.size(), index = -1;

            // Find two element which can be merged and have high priority
            for (int i = 0; i < splitedText.size() - 1; i++)
            {
                std::string tempStr = splitedText.at(i) + " " + splitedText.at(i + 1);
                if (mergeMap.find(tempStr) != mergeMap.end())
                {
                    if (mergeIndex > mergeMap[tempStr])
                    {
                        mergeIndex = mergeMap[tempStr];
                        index = i;
                    }
                }
            }

            // If there is not any merge left
            if (index == -1)
                break;

            std::vector<std::string> tempText;
            int i = 0;
            std::string targetString = splitedText.at(index) + " " + splitedText.at(index + 1);

            // merge all the token
            while (i < splitedText.size() - 1)
            {
                if (splitedText.at(i) == splitedText.at(index) && splitedText.at(i + 1) == splitedText.at(index + 1))
                {
                    tempText.push_back(splitedText.at(i) + splitedText.at(i + 1));
                    i++;
                }
                else
                    tempText.push_back(splitedText.at(i));
                i++;
            }
            if (i == splitedText.size() - 1)
                tempText.push_back(splitedText.at(i));
            splitedText = std::move(tempText);
        }

        std::vector<int> res(splitedText.size() + 1, 0);
        res[0] = 128000;

        // Convert to token to index
        for (int i = 0; i < splitedText.size(); i++)
            res[i + 1] = vocabMap[splitedText[i]];
        grid<int> encRes = VectorToGrid(res);
        return encRes;
    }

    std::string decode(grid<int> encd)
    {
        std::string decdString = "";
        for (int i = 0; i < encd.size; i++)
        {
            decdString += vocabArr[encd(i)];
        }

        size_t pos;

        while ((pos = decdString.find("Ġ")) != std::string::npos)
            decdString.replace(pos, 2, " ");

        while ((pos = decdString.find("Ċ")) != std::string::npos)
            decdString.replace(pos, 2, "\n");
        return decdString;
    }
};
