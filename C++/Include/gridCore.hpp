#pragma once

#include <iostream>

template <typename T>
class gridCore
{
T* arr;
int* stride;
int *shape;
int dim;
int size;
public:
    gridCore(std::initializer_list<int> list, T val) : size(1), dim(list.size())
    {
        stride = new int[list.size()];
        shape = new int[list.size()];

        std::copy(list.begin(), list.end(), shape);

        // Initialize stride
        int temp = 1;
        for(int i = 0; i < dim; i++)
        {
            // stride is number of jumps from one dim to next
            stride[dim - i - 1] = temp; 
            temp *= shape[dim - i - 1];
            size *= shape[i];
        }

        arr = new T[size];
        std::fill(arr, arr + size, val);
    }

    ~gridCore()
    {
        delete[] arr;
        delete[] stride;
        delete[] shape;
    }

    void Print()
    {
        std::cout << size << " " << dim << std::endl;
        for(int i = 0; i < dim; i++)
            std::cout << stride[i] << " ";
        std::cout << std::endl;
        for(int i = 0; i < dim; i++)
            std::cout << shape[i] << " ";
        std::cout << std::endl;
        for(int i = 0; i < size; i++)
            std::cout << arr[i] << " ";
        std::cout << std::endl;
    }
};