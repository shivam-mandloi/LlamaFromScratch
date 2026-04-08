#include <iostream>
#include "tokenizer.hpp"
// using namespace std;

int main()
{
    tokenizer tkn;

    std::string a = "This is test string";
    at::Tensor encdInpt = tkn.encode(a);

    return 0;
}