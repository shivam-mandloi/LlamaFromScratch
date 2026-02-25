#pragma once

#include "gridCore.hpp"

template<typename T, typename ...Args>
void permute(grid<T> &val, Args ...args)
{
    int indeces[] = {args...};
    size_t *newStride = new size_t[val.dim], *newShape = new size_t[val.dim];
    for(int i = 0; i < val.dim; i++)
    {
        newStride[i] = val.stride[indeces[i]];
        newShape[i] = val.shape[indeces[i]];
    }

    delete[] val.stride;
    delete[] val.shape;

    val.stride = newStride;
    val.shape = newShape;
}

template<typename T, typename ...Args>
void view(grid<T> &val, Args ...args)
{
    
}