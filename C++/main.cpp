#include "grid.hpp"

using namespace std;

int main()
{
    grid<int> arr({3, 4, 2}, 4);
    
    arr(0, 1, 1) = -1;
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


    grid<int> highDimTensor({3, 2, 3, 4}, 1);
    highDimTensor(1, 0, 2, 2) = 10;
    print(highDimTensor);
    permute(highDimTensor, 0, 1, 3, 2);

    print(highDimTensor);

    grid<float> vec({5}, 1.0);
    print(vec);
    view(vec, 5, 1); // view can also used as squeeze and unsqueeze
    print(vec);
    view(vec, 5);
    print(vec);

}
