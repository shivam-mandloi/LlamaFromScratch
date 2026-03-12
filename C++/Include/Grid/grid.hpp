#pragma once

#include "gridCore.hpp"
#include <cblas.h>
#include <algorithm>
#include <string>
#include <fstream>

/*
    Basic
*/
template <typename T, typename... Args>
void permute(grid<T> &val, Args... args)
{
    int indices[] = {args...};
    size_t *newStride = new size_t[val.dim], *newShape = new size_t[val.dim];
    for (int i = 0; i < val.dim; i++)
    {
        newStride[i] = val.stride[indices[i]];
        newShape[i] = val.shape[indices[i]];
    }

    delete[] val.stride;
    delete[] val.shape;

    val.stride = newStride;
    val.shape = newShape;
}

template <typename T, typename... Args>
void view(grid<T> &val, Args... args)
{
    int indices[] = {args...};
    int size = sizeof(indices) / sizeof(indices[0]);

    delete[] val.shape;
    delete[] val.stride;

    val.shape = new size_t[size];
    val.stride = new size_t[size];
    val.dim = size;

    for (int i = 0; i < size; i++)
        val.shape[i] = indices[i];
    int op = 1;
    for (int i = size - 1; i > -1; i--)
    {
        val.stride[i] = op;
        op *= val.shape[i];
    }
}

template <typename T>
void ContiguousHF(grid<T> &arr, T *newArr, int *index, int dim, int &actualIndex)
{
    if (arr.dim == dim)
    {
        newArr[actualIndex++] = arr.get(index);
        return;
    }

    for (int i = 0; i < arr.shape[dim]; i++)
    {
        index[dim] = i;
        ContiguousHF(arr, newArr, index, dim + 1, actualIndex);
    }
}

template <typename T>
void contiguous(grid<T> &arr)
{
    int *index = new int[arr.dim];
    T *newArr = new T[arr.size];
    std::fill(index, index + arr.dim, 0);
    int dim = 0, actualIndex = 0;
    ContiguousHF(arr, newArr, index, dim, actualIndex);

    delete[] arr.arr;
    delete[] index;

    arr.arr = newArr;
    int size = 1;
    for (int i = arr.dim - 1; i >= 0; i--)
    {
        arr.stride[i] = size;
        size *= arr.shape[i];
    }
}

template <typename T>
void printHF(grid<T> &arr, int dim, int *index)
{
    if (dim == arr.dim)
    {
        std::cout << arr.get(index) << " ";
        return;
    }

    for (int i = 0; i < arr.shape[dim]; i++)
    {
        index[dim] = i;
        printHF(arr, dim + 1, index);
    }
    std::cout << std::endl;
}

template <typename T>
void print(grid<T> &arr)
{
    int dim = 0;
    int *index = new int[arr.dim];
    std::fill(index, index + arr.dim, 0);

    printHF(arr, dim, index);
    delete[] index;
}

/*
    Advance
*/

template <typename T>
void MultiplicationHF(grid<T> &a,
                      grid<T> &b,
                      grid<T> &c,
                      int *index1,
                      int *index2,
                      int *index3,
                      int dim,
                      int &m1,
                      int &n1,
                      int &m2,
                      int &n2,
                      double *arr1,
                      double *arr2,
                      double *res)
{
    /*
        Parameters:
            - First tensor
            - Second tensor
            - Result tensor
            - Current Index array of first tensor
            - Current Index array of second tensor
            - Current Index array of result tensor
            - Current dimension
            - size of first matrix (m1 X n1)
            - size of second matrix (m2 X n2)
            - Temporary memory for matrix from grid a
            - Temporary memory for matrix from grid b
            - Temporary memory for matrix from grid c
    */
    if (dim + 2 == a.dim)
    {
        // Get actual index in continuous array, using virtual index
        int startPntrForFirst = a.GetIndex(index1);
        int startPntrForSecond = b.GetIndex(index2);

        // Copy matrix a and b to arr1 and arr2,
        std::copy(a.arr + startPntrForFirst, a.arr + startPntrForFirst + (m1 * n1), arr1);
        std::copy(b.arr + startPntrForSecond, b.arr + startPntrForSecond + (m2 * n2), arr2);

        // Perform matrix multiplication
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m1,
            n2,
            n1,
            1.0,
            arr1,
            n1,
            arr2,
            n2,
            0.0,
            res,
            n2);

        // If Grids are float type
        // cblas_sgemm (
        //     CblasRowMajor,
        //     CblasNoTrans,
        //     CblasNoTrans,
        //     m1,
        //     n2,
        //     n1,
        //     1.0,
        //     arr1,
        //     n1,
        //     arr2,
        //     n2,
        //     0.0,
        //     res,
        //     n2);

        int startCrntIndexRes = c.GetIndex(index3);

        // Copy result in grid c
        for (int i = 0; i < (m1 * n2); i++)
            c.arr[startCrntIndexRes + i] = res[i];

        return;
    }

    // Get the highest shape at respective dim
    int maxSizeInDim = std::max(a.shape[dim], b.shape[dim]);
    for (int i = 0; i < maxSizeInDim; i++)
    {
        // Broadcast last matrix of smaller shape matrix on dim
        // Counter multiplication of 3d or more dimenson tensor with >3d tensor
        index1[dim] = (a.shape[dim] == 1) ? 0 : i;
        index2[dim] = (b.shape[dim] == 1) ? 0 : i;
        index3[dim] = i;

        MultiplicationHF(a, b, c, index1, index2, index3, dim + 1, m1, n1, m2, n2, arr1, arr2, res);
    }
}

template <typename T>
grid<T> Multiplication(grid<T> &a, grid<T> &b, bool isContiguous = false)
{
    /*
        c = a @ b
            Here @ is matrix multiplication

        Note:
            - Only define for the double
            - Function expect same dim for both a and b
            - Only for dim >= 2, need to convert vector (1 X n) or (n X 1)
            - It Will handle automatically case where the shape is 1 for one matrix, by broadcasting it to other matrix size
            - Third point (Above point) never check if multiplication is right it can also multiply the case for example (4, 1, 3, 2) X (1, 3, 2, 4)
            - Only for float or double type grid, needs to uncomment based on type of grid
    */
    if (a.dim != b.dim || a.dim < 2)
    {
        // std::cout << a.dim << " " << b.dim << " " << a.size << " " << b.size << std::endl;
        std::cout << "Matrix Multiplication shape or dim not match" << std::endl;
        exit(0);
    }

    // update matrix a and b
    if (!isContiguous)
    {
        contiguous(a);
        contiguous(b);
    }

    // initialize the all three tensor index
    int *index1 = new int[a.dim]();
    int *index2 = new int[a.dim]();
    int *index3 = new int[a.dim]();

    // Get dimension of target tensor
    std::vector<int> dimOfTargetGrid(a.dim);

    for (int i = 0; i < a.dim - 2; i++)
        dimOfTargetGrid[i] = std::max(a.shape[i], b.shape[i]);

    int m1 = a.shape[a.dim - 2], n1 = a.shape[a.dim - 1], m2 = b.shape[a.dim - 2], n2 = b.shape[a.dim - 1];

    // Set the last 2 dim of target grid
    dimOfTargetGrid[a.dim - 2] = m1;
    dimOfTargetGrid[a.dim - 1] = n2;

    // Initialization of target grid
    grid<T> c(dimOfTargetGrid, 0);

    // Create raw array for the two matrix
    T *arr1 = new T[m1 * n1];
    T *arr2 = new T[m2 * n2];

    // Intialize the matrix to store result
    T *res = new T[m1 * n2];

    // Call helping function
    MultiplicationHF(a, b, c, index1, index2, index3, 0, m1, n1, m2, n2, arr1, arr2, res);

    // free memory
    delete[] arr1;
    delete[] arr2;
    delete[] res;
    delete[] index1;
    delete[] index2;
    delete[] index3;

    return c;
}

template <typename T>
grid<T> Addition(grid<T> &a, grid<T> &b, bool isContiguous = false)
{
    /*
        Note:
            - Does not check if all the dimension size match
            - Is grid same for virtual and actual
    */
    if (a.dim != b.dim || a.size != b.size)
    {
        std::cout << "Matrix Multiplication shape or dim not match" << std::endl;
        exit(0);
    }

    if (!isContiguous)
    {
        contiguous(a);
        contiguous(b);
    }

    grid<T> c(b);

    cblas_daxpy(a.size, 1.0, a.arr, 1, c.arr, 1);
    return c;
}

template <typename T>
void ScalarMul(grid<T> &a, double scalar)
{
    cblas_dscal(a.size, scalar, a.arr, 1);
}

// std::vector<NeuroVec<double>> ReadTxtFile(std::string path)
// {
//     std::fstream newFile;
//     std::string temp;
//     std::vector<NeuroVec<double>> res;
//     newFile.open(path, std::ios::in);
//     if (!newFile.is_open())
//     {
//         std::cerr << "Error: Could not open file " << path << std::endl;
//         exit(0);
//     }
    
//     while (getline(newFile, temp))
//     {
//         if (temp != "")
//             res.push_back(ConvertVectorToNeuroVec(SplitString(temp)));
//     }
//     return res;
// }