#include "grid.hpp"
#include <iostream>
#include <cblas.h>


using namespace std;

int main()
{
    // int M = 2;
    // int K = 3;
    // int N = 2;

    // double A[] = {
    //     1,2,3,
    //     4,5,6
    // };

    // double B[] = {
    //     7,8,
    //     9,10,
    //     11,12
    // };

    grid<double> C({2, 2}, 0);

    grid<double> A({2, 3}, 2);
    grid<double> B({3,2},3);    

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        A.shape[0],
        B.shape[1],
        A.shape[1],
        1.0,
        A.arr,
        A.shape[1],
        B.arr,
        B.shape[1],
        0.0,
        C.arr,
        B.shape[1]
    );

    print(C);
}