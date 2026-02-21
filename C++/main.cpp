#include "gridCore.hpp"

using namespace std;

int main()
{
    gridCore<int> gdCore({2, 2, 3}, 4);
    gdCore.Print();
    std::cout << gdCore(0, 0, 1) << std::endl;

    gridCore<int> anotherOne(gdCore);
    anotherOne = gdCore;
    std::cout << anotherOne(0, 0, 1) << std::endl;
    std::cout << gdCore.size << " " << gdCore.dim << std::endl;

}