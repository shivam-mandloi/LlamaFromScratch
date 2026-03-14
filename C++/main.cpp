#include "tokenizer.hpp"

using namespace std;

int main()
{
    auto start = std::chrono::steady_clock::now();
    // auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // totalTime.count()

    tokenizer tkn;
    grid<int> encdInpt = tkn.encode("Context:\nMy name is shivam mandloi. I am a MTech(AI) student at IISc banglore, I love to write a code and learn machine leanring. I use C++/python to write code. \nInstruction:\nYou are a AI assistant of person whos informtion given in Context section.\nQuestion:\nWho are you, what realtion you share with shivam mandloi?\n");
    auto mid = std::chrono::steady_clock::now();
    print(encdInpt);
    cout << tkn.decode(encdInpt) << std::endl;
    auto end = std::chrono::steady_clock::now();

    auto a1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto a2 = std::chrono::duration_cast<std::chrono::milliseconds>(mid - start);
    auto a3 = std::chrono::duration_cast<std::chrono::milliseconds>(end - mid);

    cout << "Total Time: " << a1.count() << "ms |  Encrypt Time: " << a2.count() << "ms | Decrypted Time " << a3.count() << endl;
    return 0;
}