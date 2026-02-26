#include "grid.hpp"

using namespace std;

int main()
{
    grid<int> arr({3, 4, 2}, 4);
    
    arr[0, 1, 1] = -1;
    print(arr);

    view(arr, 3, 8);    
    print(arr);
    arr[1, 7] = -10;
    print(arr);

    
    permute(arr, 1, 0);
    print(arr);
    contiguous(arr);
    view(arr, 4, 2, 3);
    cout << "After contiguous" << endl;
    print(arr);

}
