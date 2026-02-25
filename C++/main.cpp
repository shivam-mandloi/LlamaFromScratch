#include "grid.hpp"

using namespace std;

int main()
{
    grid<int> gdCore({2, 2, 3}, 4);
    gdCore.Print();
    std::cout << gdCore(0, 0, 1) << std::endl;

    grid<int> anotherOne(gdCore); // copy gdCore variable to anotherOne
    anotherOne = gdCore; // not copy transfer from gdCore to anotherOne
    anotherOne[0, 0, 1] = 1;
    std::cout << anotherOne(0, 0, 1) << std::endl;
    anotherOne.Print();

    
    // Example of Permute
    grid<float> floatArr({3, 2}, 0);
    floatArr[1,1] = -1;
    floatArr[0,1] = -2;
    cout << endl;
    for(int i = 0; i < floatArr.shape[0]; i++)
    {
        for(int j = 0; j < floatArr.shape[1]; j++)
        {
            cout << floatArr(i, j) << " ";
        }
        cout << endl;
    }
    cout << endl;
    permute(floatArr, 1, 0);
    for(int i = 0; i < floatArr.shape[0]; i++)
    {
        for(int j = 0; j < floatArr.shape[1]; j++)
        {
            cout << floatArr(i, j) << " ";
        }
        cout << endl;
    }
}
