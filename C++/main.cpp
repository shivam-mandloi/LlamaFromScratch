#include <iostream>
#include "llama.hpp"

int main()
{
    tokenizer tkn;

    std::string a = "This is test string";
    at::Tensor encdInpt = tkn.encode(a);

    return 0;
}