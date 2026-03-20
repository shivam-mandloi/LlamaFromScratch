#pragma once

#include "grid.hpp"
#include "HelpingFunction.hpp"

#include <thread>

class Attention
{
    size_t qkvDiml;
    grid<float> query, key, value, output;

public:
    // 1 for the batch
    Attention(int index, size_t _qkvDim) : qkvDiml(_qkvDim), query({1, 4096, 4096}, 0.0), key({1, 4096, 1024}, 0.0), value({1, 4096, 1024}, 0.0), output({1, 4096, 4096}, 0.0)
    {
        // initialize the size of array
        // size_t qoArrSize = 4096 * 4096, kvArrSize = 4096 * 1024;
        // float *qArr = new float[qoArrSize], *oArr = new float[qoArrSize], *kArr = new float[kvArrSize], *vArr = new float[kvArrSize];

        // Load query parameter
        LoadBin("model.layers." + std::to_string(index) + ".self_attn.q_proj.weight.bin", query.arr, query.size);
        LoadBin("model.layers." + std::to_string(index) + ".self_attn.k_proj.weight.bin", key.arr, key.size);
        LoadBin("model.layers." + std::to_string(index) + ".self_attn.v_proj.weight.bin", value.arr, value.size);
        LoadBin("model.layers." + std::to_string(index) + ".self_attn.o_proj.weight.bin", output.arr, output.size);
    }

    void forward(grid<float> &x, grid<float> &mask)
    {
        /*
            x: B X seq X n
            mask: B X seq X seq

            seq = Sequence Size (highest among batch) | n = Input Dim (for llama 4096) | B = Batch Size
        */

        size_t seqLen = x.shape[1], batchSize = x.shape[0];

        permute(x, 0, 2, 1); // (batch X seq X n) -> (batch X n X seq)
        grid<float> q = Multiplication(query, x);
        grid<float> k = Multiplication(query, x);
        grid<float> v = Multiplication(query, x);

        view(q, batchSize, seqLen, 32, 128);
        view(k, batchSize, seqLen, 8, 128);
        view(v, batchSize, seqLen, 8, 128);

        std::cout << q.shape[0] << " " << q.shape[1] << " " << q.shape[2] << " " << q.shape[3] << std::endl;
        std::cout << k.shape[0] << " " << k.shape[1] << " " << k.shape[2] << " " << q.shape[3] << std::endl;
        std::cout << v.shape[0] << " " << v.shape[1] << " " << v.shape[2] << " " << q.shape[3] << std::endl;
        // std::cout << q.size << " "  << k.size << " " << v.size << " " << output.size << std::endl;
    }
};