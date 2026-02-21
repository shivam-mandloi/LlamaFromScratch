#pragma once

#include <iostream>

template <typename T>
class gridCore
{
    T *arr;
    size_t size, dim;
    size_t *stride, *shape;

public:
    gridCore(std::initializer_list<int> list, T val) : dim(list.size())
    {
        stride = new size_t[list.size()];
        shape = new size_t[list.size()];

        std::copy(list.begin(), list.end(), shape);

        // Initialize stride
        size = 1;
        for (int i = dim - 1; i >= 0; --i)
        {
            stride[i] = size;
            size *= shape[i];
        }

        arr = new T[size];
        std::fill(arr, arr + size, val);
    }

    ~gridCore() noexcept
    {
        delete[] arr;
        delete[] stride;
        delete[] shape;
    }

    gridCore(const gridCore<T> &other)
    {
        dim = other.dim;
        size = other.size;

        arr = new T[size];
        stride = new size_t[dim];
        shape = new size_t[dim];

        std::copy(other.arr, other.arr + size, arr);
        std::copy(other.stride, other.stride + dim, stride);
        std::copy(other.shape, other.shape + dim, shape);
    }

    // This function move one grid to other gird
    // This function does not copy
    gridCore<T> &operator=(gridCore<T> &other)
    {
        if (this != &other)
        {
            delete[] arr;
            delete[] stride;
            delete[] shape;

            shape = other.shape;
            arr = other.arr;
            stride = other.stride;
            dim = other.dim;
            size = other.size;

            other.arr = nullptr;
            other.shape = nullptr;
            other.stride = nullptr;
            other.dim = 0;
            other.size = 0;
        }
        return *this;
    }

    int GetIndex(std::initializer_list<int> &list)
    {
        if (list.size() != dim)
        {
            std::cerr << "Dim misMatch got " << std::endl;
            exit(0);
        }
        int index = 0, i = 0;
        for (int val : list)
        {
            index += (val * stride[i]);
            i += 1;
        }
        return index;
    }

    inline T &operator[](int index) { return arr[index]; }

    template <typename... Args>
    inline T &operator()(Args... args)
    {
        int indices[] = {args...};
        int index = 0;
        for (int i = 0; i < dim; ++i)
        {
            index += indices[i] * stride[i];
        }
        return arr[index];
    }

    void Print()
    {
        std::cout << size << " " << dim << std::endl;
        for (int i = 0; i < dim; i++)
            std::cout << stride[i] << " ";
        std::cout << std::endl;
        for (int i = 0; i < dim; i++)
            std::cout << shape[i] << " ";
        std::cout << std::endl;
        for (int i = 0; i < size; i++)
            std::cout << arr[i] << " ";
        std::cout << std::endl;
    }
};