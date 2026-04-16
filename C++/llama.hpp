#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <ATen/ATen.h>

#include "HelpingFunction.hpp"

void Softmax(at::Tensor &input)
{
    // Only Create for the Attention, not generalized version
    /*
        input: (batch, 24, s, seqLen)
        output: (batch, 24, s, seqLen)
    */

    // argument: Tensor, dim, keepDim
    auto result = at::max(input, -1, true);
    at::Tensor maxVal = std::get<0>(result);

    input -= maxVal;
    input.exp_();

    at::Tensor sumVal = input.sum(-1, true);
    input /= sumVal;
}

void RMSNorm(at::Tensor &x, at::Tensor &w, float eps = 1e-5)
{
    // Only Create for specific purpose, not generalized version
    /*
        x: b X s X n
        w: n
        b = batch size | s = sequence | n = input dim
    */
    x = x.pow(2).mean(-1, true);
    at::Tensor rms = at::sqrt(x + eps);
    x = (x / rms) * w;
}

at::Tensor CreateMask(int tSeq, int s)
{
    /*
        -> tril version
        -> t here is total sequence len, means  tSeq = seq + s
        output: 1 X 1 X s X tSeq
    */
    auto full = at::tril(at::ones({tSeq, tSeq}));

    return full.slice(0, tSeq - s, tSeq).unsqueeze(0).unsqueeze(0);
}

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
    /*
        kCache: b X s X h X n
        vCache: b X s X h X n

        b = batch size | s = sequence size | h = number of head | n = qkvDim
    */
    at::Tensor query, key, value, output, kCache, vCache;
    int qkvDim = 128;
    float qkvDimRoot = 1.0 / std::sqrt(qkvDim);

public:
    Attention(int index, std::vector<int64_t> querySize,
              std::vector<int64_t> keySize,
              std::vector<int64_t> valueSize,
              std::vector<int64_t> outputSize,
              int batchSize,
              int maxSeqLen)
    {
        // Load Parameters
        kCache = at::zeros({batchSize, maxSeqLen, 8, 128});
        vCache = at::zeros({batchSize, maxSeqLen, 8, 128});

        query = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.q_proj.weight.bin", querySize);
        key = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.k_proj.weight.bin", keySize);
        value = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.v_proj.weight.bin", valueSize);
        output = LoadTensor("model.layers." + std::to_string(index) + ".self_attn.o_proj.weight.bin", outputSize);
    }

    at::Tensor forward(at::Tensor &inp,
                       std::vector<float> &sinTheta,
                       std::vector<float> &cosTheta,
                       int seqLen)
    {
        /*
            x: (b X s X n) | (b X 1 X n)
            ropeTheta: (qkvDim / 2)
            sinTheta: virtualy size (batch size X (qkvDim/2))
            cosTheta: virtualy size (batch size X (qkvDim/2))
            seqlen: sequence size

            For sinTheta and cosTheta have higher or equal size to sequence length
        */

        // Get the bath, seqLen and dim info
        int64_t b = inp.size(0), s = inp.size(1), n = inp.size(2);

        // Dim: b X s X (head * qkvDim)
        at::Tensor q = inp.matmul(query.t());
        at::Tensor k = inp.matmul(key.t());
        at::Tensor v = inp.matmul(value.t());

        // Dim: b X s X head X qkvDim/2 X 2
        q = q.contiguous().view({b, s, 24, qkvDim / 2, 2});
        v = v.contiguous().view({b, s, 8, qkvDim});
        k = k.contiguous().view({b, s, 8, qkvDim / 2, 2});

        /*
            ROPE Implementation
        */

        // Dim: 1 X s X 1 X (qkvDim/2)
        // generate the last sin and cos tensor from the vector
        at::Tensor sin = at::from_blob(sinTheta.data() + static_cast<int>((seqLen * (qkvDim / 2))), {1, s, 1, static_cast<int64_t>(qkvDim / 2)});
        at::Tensor cos = at::from_blob(cosTheta.data() + static_cast<int>((seqLen * (qkvDim / 2))), {1, s, 1, static_cast<int64_t>(qkvDim / 2)});

        // Generate the sin and cos value for the query
        at::Tensor tSin = sin.expand({b, s, 24, sin.size(-1)});
        at::Tensor tCos = cos.expand({b, s, 24, cos.size(-1)});

        auto x = q.select(-1, 0);
        auto y = q.select(-1, 1);

        at::Tensor newX = x * tCos - y * tSin;
        at::Tensor newY = x * tSin + y * tCos;

        q.select(-1, 0).copy_(newX);
        q.select(-1, 1).copy_(newY);

        q = q.view({b, s, 24, qkvDim});

        // Generate the sin and cos value for the key
        tSin = sin.expand({b, s, 8, sin.size(-1)});
        tCos = cos.expand({b, s, 8, cos.size(-1)});

        x = k.select(-1, 0);
        y = k.select(-1, 1);

        newX = x * tCos - y * tSin;
        newY = x * tSin + y * tCos;

        k.select(-1, 0).copy_(newX);
        k.select(-1, 1).copy_(newY);

        k = k.view({b, s, 8, qkvDim});

        // update the Key and value cache
        // Add s new elements in sequence
        kCache.slice(1, seqLen, seqLen + s).copy_(k);
        vCache.slice(1, seqLen, seqLen + s).copy_(v);

        // Make key and value to (batch X seqLen X 24 X qkvDim)
        // repeat_interleave(a, b), repeat index b, by a times
        // Slice first seqLen + 1 vectors from the cache
        k = kCache.slice(1, 0, seqLen + s).unsqueeze(3).repeat_interleave(3, 3).reshape({b, seqLen + s, 24, qkvDim});
        v = vCache.slice(1, 0, seqLen + s).unsqueeze(3).repeat_interleave(3, 3).reshape({b, seqLen + s, 24, qkvDim});

        // Dim: (batch X 24 X s X qkvDim)
        q = q.permute({0, 2, 1, 3});

        // Dim: (batch X 24 X qkvDim X (seqLen + s))
        k = k.permute({0, 2, 3, 1});

        // (batch X 24 X (seqLen + s) X qkvDim)
        v = v.permute({0, 2, 1, 3});

        // Dim: (batch X 24 X s X (seqLen + s))
        at::Tensor score = q.matmul(k) * qkvDimRoot;

        // Used when input is not single input, but sequence
        // Example when we are providing context or question
        if (s > 1)
        {
            at::Tensor mask = CreateMask(seqLen + s, s);
            score = score.masked_fill(mask == 0, -std::numeric_limits<float>::infinity());
        }

        // (batch X 24 X s X (seqLen + s))
        Softmax(score);

        // (batch X 24 X s X qkvDim)
        v = score.matmul(v).permute({0, 2, 1, 3}).contiguous().view({b, s, 24 * qkvDim});

        return v.matmul(output.t());
    }
};

class MLP
{
    at::Tensor down, up, gate;

public:
    MLP(int index,
        std::vector<int64_t> downSize,
        std::vector<int64_t> upSize,
        std::vector<int64_t> gateSize)
    {
        // n: Input Dim
        // [n, 8192]
        down = LoadTensor("model.layers." + std::to_string(index) + ".mlp.down_proj.weight.bin", downSize);
        // [8192, n]
        up = LoadTensor("model.layers." + std::to_string(index) + ".mlp.up_proj.weight.bin", upSize);
        // [8192, n]
        gate = LoadTensor("model.layers." + std::to_string(index) + ".mlp.gate_proj.weight.bin", gateSize);
    }

    at::Tensor forward(at::Tensor &x)
    {
        /*
            x: b X s X n
            b = batch size | s = sequence | n = input dim
        */

        gate = gate.matmul(x.permute({0, 2, 1}));
        up = up.matmul(x.permute({0, 2, 1}));

        gate *= at::sigmoid(gate);

        gate = gate * up;

        return down.matmul(gate);
    }
};

class Transformer
{
    at::Tensor inputRMSNorm, postAttnRMSNorm;
    Attention attn;
    MLP mlp;

public:
    Transformer(int index,
                std::vector<int64_t> querySize,
                std::vector<int64_t> keySize,
                std::vector<int64_t> valueSize,
                std::vector<int64_t> outputSize,
                std::vector<int64_t> downSize,
                std::vector<int64_t> upSize,
                std::vector<int64_t> gateSize,
                std::vector<int64_t> inputRMSNormSize,
                std::vector<int64_t> postAttnRMSNormSize,
                int batchSize,
                int maxSeqLen) : attn(index, querySize, keySize, valueSize, outputSize, batchSize, maxSeqLen),
                                 mlp(index, downSize, upSize, gateSize)
    {
        inputRMSNorm = LoadTensor("model.layers." + std::to_string(index) + ".input_layernorm.weight.bin", downSize);
        postAttnRMSNorm = LoadTensor("model.layers." + std::to_string(index) + ".post_attention_layernorm.weight.bin", downSize);
    }

    void forward(at::Tensor &inp,
                 std::vector<float> &sinTheta,
                 std::vector<float> &cosTheta,
                 int seqLen,
                 at::Tensor mask = at::Tensor())
    {
        /*
            inp: b X s X n
            b = batch size | s = sequence | n = input dim
        */

        at::Tensor attnInp = inp.clone();
        RMSNorm(attnInp, inputRMSNorm);

        // call when mask has not define
        if (!mask.defined())
            attnInp = attn.forward(inp, sinTheta, cosTheta, seqLen);
        // else

        inp = inp + attnInp;
        attnInp = inp.clone();

        RMSNorm(attnInp, postAttnRMSNorm);
        attnInp = mlp.forward(attnInp);

        inp += attnInp;
    }
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

        theta = at::from_blob(thetaData.data(), {
                                                    static_cast<int64_t>(qkvDim / 2),
                                                })
                    .clone();
    }
};