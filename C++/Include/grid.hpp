#pragma once

#include "gridCore.hpp"

template<typename T, typename ...Args>
void permute(grid<T> &val, Args ...args)
{
    int indices[] = {args...};
    size_t *newStride = new size_t[val.dim], *newShape = new size_t[val.dim];
    for(int i = 0; i < val.dim; i++)
    {
        newStride[i] = val.stride[indices[i]];
        newShape[i] = val.shape[indices[i]];
    }

    delete[] val.stride;
    delete[] val.shape;

    val.stride = newStride;
    val.shape = newShape;
}

template<typename T, typename ...Args>
void view(grid<T> &val, Args ...args)
{
    int indices[] = {args...};
    int size = sizeof(indices) / sizeof(indices[0]);
    
    delete[] val.shape;
    delete[] val.stride;

    val.shape = new size_t[size];
    val.stride = new size_t[size];
    val.dim = size;
    
    for(int i = 0; i < size; i++)    
        val.shape[i] = indices[i];
    int op = 1;
    for(int i = size-1; i > -1; i--)
    {
        val.stride[i] = op;
        op *= val.shape[i];
    }
}

template<typename T>
void ContiguousHF(grid<T> &arr, T *newArr, int* index, int dim, int &actualIndex)
{
    if(arr.dim == dim)
    {
        newArr[actualIndex] = arr.get(index);        
        return;
    }

    for(int i = 0; i < arr.shape[dim]; i++)
    {
        index[dim] = i;
        ContiguousHF(arr, newArr, index, dim + 1, actualIndex);
        actualIndex += 1;
    }
    actualIndex-=1;
}

template<typename T>
void contiguous(grid<T> &arr)
{
    int *index = new int[arr.dim];
    T *newArr = new T[arr.size];
    std::fill(index, index + arr.dim, 0);
    int dim = 0, actualIndex = 0;
    ContiguousHF(arr, newArr, index, dim, actualIndex);

    delete[] arr.arr;
    arr.arr = newArr;
    int size = 1;
    for (int i = arr.dim - 1; i >= 0; i--)
    {
        arr.stride[i] = size;
        size *= arr.shape[i];
    }
}


template<typename T>
void printHF(grid<T> &arr, int dim, int *index)
{
    if(dim == arr.dim)
    {
        std::cout << arr.get(index) << " ";
        return ;
    }    

    for(int i = 0; i < arr.shape[dim]; i++)
    {
        index[dim] = i;
        printHF(arr, dim + 1, index);
    }
    std::cout << std::endl;
}

template<typename T>
void print(grid<T> &arr)
{
    int dim = 0;
    int *index = new int[arr.dim];
    std::fill(index, index + arr.dim, 0);

    printHF(arr, dim, index);
}