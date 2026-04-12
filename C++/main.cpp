#include <iostream>
#include <chrono>
#include "llama.hpp"

int main()
{
    std::vector<int64_t> qSize = {3072, 3072}, kSize = {1024, 3072}, vSize = {1024, 3072}, oSize = {3072, 3072};
    std::vector<int64_t> size = {50257, 768};
    std::cout << "Start" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    at::Tensor a = LoadTensor("model.embed_tokens.weight.bin", size);
    for(int i = 0; i < 28; i++)
    {
        Attention attn(i, qSize, kSize, vSize, oSize);
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Total Time: " << duration.count() << std::endl;
    return 0;
}