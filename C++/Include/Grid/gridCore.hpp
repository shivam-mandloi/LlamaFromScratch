#pragma once

#include <iostream>
#include <vector>

template <typename T>
class grid
{
public:
    T *arr;
    size_t size, dim;
    size_t *stride, *shape;

    grid()
    {
        arr = nullptr;
        size = 0;
        dim = 0;
        stride = nullptr;
        shape = nullptr;
    }

    grid(std::initializer_list<int> list, T val) : dim(list.size())
    {
        stride = new size_t[list.size()];
        shape = new size_t[list.size()];

        std::copy(list.begin(), list.end(), shape);

        // Initialize stride
        size = 1;
        for (int i = dim - 1; i >= 0; i--)
        {
            stride[i] = size;
            size *= shape[i];
        }

        arr = new T[size];
        std::fill(arr, arr + size, val);
    }

    grid(std::vector<int> list, T val) : dim(list.size())
    {
        stride = new size_t[list.size()];
        shape = new size_t[list.size()];

        std::copy(list.begin(), list.end(), shape);

        // Initialize stride
        size = 1;
        for (int i = dim - 1; i >= 0; i--)
        {
            stride[i] = size;
            size *= shape[i];
        }

        arr = new T[size];
        std::fill(arr, arr + size, val);
    }

    ~grid() noexcept
    {
        delete[] arr;
        delete[] stride;
        delete[] shape;
    }

    grid(const grid<T> &other)
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
    grid<T> &operator=(grid<T> &other)
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

    // Update value at index
    template <typename... Args>
    inline T &operator[](Args... args)
    {
        int index = 0;
        int list[] = {args...};
        for (int i = 0; i < dim; i++)
            index += (list[i] * stride[i]);
        return arr[index];
    }

    // Get value at index
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

    inline T get(int *args)
    {
        int index = 0;
        for (int i = 0; i < dim; ++i)
        {
            index += args[i] * stride[i];
        }
        return arr[index];
    }

    inline void push(int *args, T val)
    {
        int index = 0;
        for (int i = 0; i < dim; i++)
            index += (args[i] * stride[i]);
        arr[index] = val;
    }

    inline int GetIndex(int *args)
    {
        int index = 0;
        for (int i = 0; i < dim; ++i)
        {
            index += args[i] * stride[i];
        }
        return index;
    }
};