#include "gridCore.hpp"

using namespace std;

int main()
{
    gridCore<int> gdCore({2, 2, 3}, 4);
    gdCore.Print();

}