#include <stdio.h>
#include "../Common/helper_math.h"


void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err) );
        exit(EXIT_FAILURE); 
    }
}

template <size_t blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, size_t tid)
{
    if (blockSize >= 64) {sdata[tid].x += sdata[tid + 32].x; sdata[tid].y += sdata[tid + 32].y; sdata[tid].z += sdata[tid + 32].z; sdata[tid].w += sdata[tid + 32].w;};
    if (blockSize >= 32) {sdata[tid].x += sdata[tid + 16].x; sdata[tid].y += sdata[tid + 16].y; sdata[tid].z += sdata[tid + 16].z; sdata[tid].w += sdata[tid + 16].w;};
    if (blockSize >= 16) {sdata[tid].x += sdata[tid + 8].x; sdata[tid].y += sdata[tid + 8].y; sdata[tid].z += sdata[tid + 8].z; sdata[tid].w += sdata[tid + 8].w;};
    if (blockSize >=  8) {sdata[tid].x += sdata[tid + 4].x; sdata[tid].y += sdata[tid + 4].y; sdata[tid].z += sdata[tid + 4].z; sdata[tid].w += sdata[tid + 4].w;};
    if (blockSize >=  4) {sdata[tid].x += sdata[tid + 2].x; sdata[tid].y += sdata[tid + 2].y; sdata[tid].z += sdata[tid + 2].z; sdata[tid].w += sdata[tid + 2].w;};
    if (blockSize >=  2) {sdata[tid].x += sdata[tid + 1].x; sdata[tid].y += sdata[tid + 1].y; sdata[tid].z += sdata[tid + 1].z; sdata[tid].w += sdata[tid + 1].w;};;
}

template <size_t blockSize, typename T>
__global__ void reduceCUDA(T* g_idata, T* g_odata, size_t n)
{
    __shared__ T sdata[blockSize];

    size_t tid = threadIdx.x;
    //size_t i = blockIdx.x*(blockSize*2) + tid;
    //size_t gridSize = blockSize*2*gridDim.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = {0.,0.,0.,0.};

    while (i < n) { sdata[tid] += g_idata[i]; i += gridSize; }
    //while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <size_t blockSize, typename T>
__global__ void reduceCUDAPopulate(
    T*          states, 
    const T*    positions,

    const T     newState,

    const float Jxx,
    const float Jyy,
    const float Jxy,
    const float DIP,
    const float tresh,

    T* g_odata, 
    
    const size_t next,
    const size_t prev,
    size_t n
)
{
    __shared__ T sdata[blockSize];
    //__shared__ T pos;

    size_t tid = threadIdx.x;
    //size_t i = blockIdx.x*(blockSize*2) + tid;
    //size_t gridSize = blockSize*2*gridDim.x;
    size_t i = blockIdx.x*(blockSize) + tid;
    size_t gridSize = blockSize*gridDim.x;
    sdata[tid] = {0.,0.,0.,0.};

    //TO-DO: concurecncy error!!!
    if (i == 0) {
        states[prev] = newState;
    }

    //TO-DO: concurecncy error!!!
    //if (tid == 0) {
    //    T pos = positions[next];
    //}

    T pos = positions[next];

    //define stuff for the calcualtions
    T r;
    T s;
    T acc;

    float distSqr;
    float invDist;
    float invDistCube;
    float invDistPenta;

    //__syncthreads();

    while (i < n) { 
        //skip the current position
        //concurency error!!!
        if (i == next) {
            i += gridSize;
            continue;
        } 

        //grad the i-th position and state
        r = positions[i] - pos;
        s = states[i];
        
        //stuff for the dipole-dipole interaction
        distSqr = (r.x * r.x)  +  (r.y * r.y)  +  (r.z * r.z);
        invDist = rsqrtf(distSqr);
        invDistCube = invDist * invDist * (invDist * DIP);  
        invDistPenta = invDistCube * invDist * invDist;

        acc.x = s.x * (invDistCube - 3 * invDistPenta * r.x * r.x) - s.y * 3 * invDistPenta * r.x * r.y - s.z * 3 * invDistPenta * r.x * r.z; 
        acc.y = s.y * (invDistCube - 3 * invDistPenta * r.y * r.y) - s.z * 3 * invDistPenta * r.y * r.z - s.x * 3 * invDistPenta * r.y * r.x; 
        acc.z = s.z * (invDistCube - 3 * invDistPenta * r.z * r.z) - s.x * 3 * invDistPenta * r.z * r.x - s.y * 3 * invDistPenta * r.z * r.y; 

        //stiff for the superexchange
        if (distSqr < tresh) {
            acc.x += s.x * Jxy; 
            acc.y += s.y * Jxy; 
            acc.z += s.z * Jxy; 
            acc.w = 0.0;
        }

        sdata[tid].x = sdata[tid].x + acc.x;
        sdata[tid].y = sdata[tid].y + acc.y;
        sdata[tid].z = sdata[tid].z + acc.z;
        i += gridSize; 
    }
    //while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();

    if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
    if (blockSize >=  512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >=  256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >=  128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


// PRE:
// dA is an array allocated on the GPU
// N <= len(dA) is a power of two (N >= BLOCKSIZE)
// POST: the sum of the first N elements of dA is returned
template<size_t blockSize, typename T>
T GPUReduction(T* dA, size_t N)
{
    T tot = 0.;
    size_t n = N;
    size_t blocksPerGrid = std::ceil((1.*n) / blockSize);

    T* tmp;
    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); checkCUDAError("Error allocating tmp [GPUReduction]");

    T* from = dA;

    blocksPerGrid   = std::ceil((1.*n) / blockSize);
    reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
    from = tmp;
    n = blocksPerGrid;

    do
    {
        blocksPerGrid   = std::ceil((1.*n) / blockSize);
        reduceCUDA<blockSize><<<blocksPerGrid, blockSize>>>(from, tmp, n);
        from = tmp;
        n = blocksPerGrid;
    } while (n > blockSize);

    if (n > 1)
        reduceCUDA<blockSize><<<1, blockSize>>>(tmp, tmp, n);

    cudaDeviceSynchronize();
    checkCUDAError("Error launching kernel [GPUReduction]");

    cudaMemcpy(&tot, tmp, sizeof(T), cudaMemcpyDeviceToHost); checkCUDAError("Error copying result [GPUReduction]");
    cudaFree(tmp);
    return tot;
}
