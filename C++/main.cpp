#include <iostream>
#include <chrono>
#include "llama.hpp"

int main()
{
    std::vector<int64_t> qSize = {3072, 3072}, kSize = {1024, 3072}, vSize = {1024, 3072}, oSize = {3072, 3072};
    std::vector<int64_t> size = {50257, 768};
    auto start = std::chrono::high_resolution_clock::now();
    std::string Instruction = "";
    
    llamma lma(1, 1024, Instruction);
    lma.call("Question: Explain me backpropagation?");
    std::cout << ">>";
    
    while(true)
    {
        std::string text = lma.GetNext();        
        if (text == "endOfText") 
        {
            std::cout << "\n\n[Finished]" << std::endl;
            break; 
        }
        std::cout << text << std::flush; 
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

    std::cout << "Total Time: " << duration.count() << std::endl;
    return 0;
}