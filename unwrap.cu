#include <iostream>
#include <cmath>
#include "includes/gpu_reduce.hpp"

#define BLOCKSIZE 1024

int main()
{
    using my_t = float4;

    size_t N = 10240;

    // Populate array
    my_t* A = (my_t*) malloc(sizeof(my_t) * N);
    for (size_t i = 0; i < N; i++)
        A[i] = 1;

    my_t* dA;
    cudaMalloc(&dA, sizeof(my_t)*N); checkCUDAError("Error allocating dA");
    cudaMemcpy(dA, A, sizeof(my_t)*N, cudaMemcpyHostToDevice); checkCUDAError("Error copying A"); 

    my_t tot = 0.;

    tot  = GPUReduction<BLOCKSIZE>(dA, N);

    std::cout << "N: " << N << std::endl;
    std::cout << "Result: " << tot << std::endl;
}