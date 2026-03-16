#include "tokenizer.hpp"

using namespace std;

int main()
{
    // auto start = std::chrono::steady_clock::now();
    // auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // totalTime.count()

    tokenizer tkn;
    grid<int> encdInpt = tkn.encode("testing is going on... ha ha ha");
    // print(encdInpt);
    cout << tkn.decode(encdInpt) << std::endl;    
    
    string filePath = "model.embed_tokens.weight.bin";

    grid<float> tknEmbd({128256, 4096}, 0);

    auto start = std::chrono::steady_clock::now();
    LoadBin(filePath, tknEmbd.arr, tknEmbd.size);
    auto end = std::chrono::steady_clock::now();

    auto a1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Time To read file: " << a1.count() << "ms" << std::endl;

    grid<float> temp({4096, 1}, 0);

    start = std::chrono::steady_clock::now();
    grid<float> res = Multiplication(tknEmbd, temp);
    end = std::chrono::steady_clock::now();
    
    a1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Time to multiply matrix: " << a1.count() << "ms" << std::endl;

    return 0;
}