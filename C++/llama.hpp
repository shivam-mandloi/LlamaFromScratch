#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <ATen/ATen.h>

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

    at::Tensor encode(std::string text)
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

        std::vector<int64_t> sizeOfEncodeVector = {
            static_cast<int64_t>(res.size()),
        };
        at::Tensor encRes = at::from_blob(res.data(), sizeOfEncodeVector, at::kInt).clone();
        return encRes;
    }

    std::string decode(at::Tensor encd)
    {
        std::string decdString = "";
        for (int i = 0; i < encd.size(0); i++)
            decdString += vocabArr[encd[i].item<int>()];

        size_t pos;

        while ((pos = decdString.find("Ġ")) != std::string::npos)
            decdString.replace(pos, 2, " ");

        while ((pos = decdString.find("Ċ")) != std::string::npos)
            decdString.replace(pos, 2, "\n");
        return decdString;
    }
};

class Attention
{
    at::Tensor query, key, value, output, kCache, vCache;
    int qkvDim = 128;

public:
    Attention(int index, std::vector<int64_t> querySize,
              std::vector<int64_t> keySize,
              std::vector<int64_t> valueSize,
              std::vector<int64_t> outputSize)
    {
        // Load Parameters
        query = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.q_proj.weight.bin", querySize);
        key = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.k_proj.weight.bin", keySize);
        value = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.v_proj.weight.bin", valueSize);
        output = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.o_proj.weight.bin", outputSize);
    }

    void forward(at::Tensor x,
                 at::Tensor mask,
                 std::vector<float> &sinTheta,
                 std::vector<float> &cosTheta,
                 int seqLen)
    {
        /*
            x: (b X n)
            ropeTheta: (qkvDim / 2)
            sinTheta: virtualy size (seqLen X (qkvDim/2))
            cosTheta: virtualy size (seqLen X (qkvDim/2))
            seqlen: sequence size

            For sinTheta and cosTheta have higher or equal size to sequence length
        */

        // Get the bath, seqLen and dim info
        int64_t b = x.size(0), n = x.size(2);

        // Dim: b X (head * qkvDim)
        at::Tensor q = x.matmul(query.t());
        at::Tensor k = x.matmul(key.t());
        at::Tensor v = x.matmul(value.t());

        q = q.view({b, 24, 128});
        v = v.view({b, 8, 128});
        k = k.view({b, 8, 128});

        /*
            ROPE Implementation
        */

        // Dim: seq X (qkvDim/2)
        // generate the last sin and cos tensor from the vector
        at::Tensor sin = at::from_blob(sinTheta.data() + static_cast<int>((seqLen * (qkvDim / 2))), {static_cast<int64_t>(qkvDim / 2)}).unsqueeze(0).unsqueeze(0);
        at::Tensor cos = at::from_blob(cosTheta.data() + static_cast<int>((seqLen * (qkvDim / 2))), {static_cast<int64_t>(qkvDim / 2)}).unsqueeze(0).unsqueeze(0);
        
        // Generate the sin and cos value for the query
        at::Tensor tSin = sin.repeat({b, 24, 1});
        at::Tensor tCos = cos.repeat({b, 24, 1});


        q = at::cat({
            q.slice(-1, 0, static_cast<int>(seqLen/2)) * tCos - q.slice(-1, static_cast<int>(seqLen/2), seqLen) * tSin,
            q.slice(-1, 0, static_cast<int>(seqLen/2)) * tSin + q.slice(-1, static_cast<int>(seqLen/2), seqLen) * tCos
        }, -1);


        // Generate the sin and cos value for the key
        
        tSin = sin.repeat({b, 8, 1});
        tCos = cos.repeat({b, 8, 1});

        q = at::cat({
            k.slice(-1, 0, static_cast<int>(seqLen/2)) * tCos - k.slice(-1, static_cast<int>(seqLen/2), seqLen) * tSin,
            k.slice(-1, 0, static_cast<int>(seqLen/2)) * tSin + k.slice(-1, static_cast<int>(seqLen/2), seqLen) * tCos
        }, -1);
    }
};

class MLP
{
};

class Transfomer
{
};

class llamma
{
    at::Tensor theta;
    int qkvDim = 128;
    float ropeTheta = 500000.0;

public:
    llamma()
    {
        std::vector<float> thetaData(qkvDim / 2, 0);
        for (int i = 0; i < thetaData.size(); i++)
            thetaData[i] = pow(ropeTheta, -2 * (i) / qkvDim);

        theta = at::from_blob(thetaData.data(), {static_cast<int64_t>(qkvDim / 2),}).clone();
    }
};