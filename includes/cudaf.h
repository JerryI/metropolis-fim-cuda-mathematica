#include "../Common/helper_math.h"

__device__ int getGlobalId() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__device__ int getLocalId() {
    return (threadIdx.y * blockDim.x) + threadIdx.x;
}

__device__ int getGroupId() {
    return blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
}

__device__ int getLocalSize() {
    return (blockDim.x * blockDim.y);
}

__global__ void calcMag(
    float4* spin,
    float4* partialField,
	const int length
)

#define PERS "float"

{
    __shared__ float4 localSums[THREADS];

    int tid = getLocalId();
    int bid = getGroupId();
    int localSize = getLocalSize();
    int gid = getGlobalId();

	if(gid < length) {

    localSums[tid].x = 2*spin[gid].x;
    localSums[tid].y = 2*spin[gid].y;
    localSums[tid].z = 2*spin[gid].z;
	
	} else {
	localSums[tid].x = 0;
    localSums[tid].y = 0;
    localSums[tid].z = 0;
	}

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (int stride = localSize/2; stride>0; stride /=2)
    {
        // Waiting for each 2x2 addition into given workgroup
        __syncthreads();

        // Add elements 2 by 2 between local_id and local_id + stride
        if (tid < stride)
            localSums[tid] += localSums[tid + stride];
    }

    __syncthreads();
    // Write result into partialSums[nWorkGroups]
    if (tid == 0) {
        partialField[bid].x = localSums[0].x;
        partialField[bid].y = localSums[0].y;
        partialField[bid].z = localSums[0].z;
    } 

}
