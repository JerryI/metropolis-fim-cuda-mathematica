inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __host__ __device__ float4 operator*(float4 a, float b)
{
    return make_float4(a.x * b, a.y  * b, a.z  * b,  a.w  * b);
}

inline __host__ __device__ void operator+=(double4 &a, double4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __host__ __device__ double4 operator-(double4 a, double4 b)
{
    return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

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

#define PERS "double"

{
    __shared__ double4 localSums[1024];

    unsigned int tid = getLocalId();
    unsigned int bid = getGroupId();
    unsigned int localSize = getLocalSize();
    unsigned int gid = getGlobalId();

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
    for (unsigned int stride = localSize/2; stride>0; stride /=2)
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

__global__ void calcFields(
    float4* spin,
    const float4* pos,

    const float4  newSpin,

    const float Jxx,
    const float Jyy,
    const float Jxy,
    const float DIP,
    const float tresh,

    float4* partialField,

    const int next,
    const int prev,
	const int length
)

{
    __shared__ double4 localSums[1024];

    unsigned int tid = getLocalId();
    unsigned int bid = getGroupId();
    unsigned int localSize = getLocalSize();
    unsigned int gid = getGlobalId();

    //Write the previous
    if(gid == 0) {
        spin[prev] = newSpin;
    }
	
	if(gid < length) {
	
    float4 nextPos = pos[next];

    float4 privatepos = pos[gid];
        
    float4 r = privatepos - nextPos;

    double distSqr = (r.x * r.x)  +  (r.y * r.y)  +  (r.z * r.z);

    if (distSqr < 0.1f) {
        localSums[tid].x = 0;
        localSums[tid].y = 0;
        localSums[tid].z = 0;
    } else {
        double invDist = rsqrtf(distSqr);
        double invDistCube = invDist * (double)DIP * invDist * invDist;  
        double invDistPenta = invDistCube * invDist * invDist;
        double4 privatefield; 
        float4 privatespin = spin[gid];
 

        privatefield.x = (double)privatespin.x * (invDistCube - 3 * invDistPenta * r.x * r.x) - (double)privatespin.y * 3 * invDistPenta * r.x * r.y - (double)privatespin.z * 3 * invDistPenta * r.x * r.z; 
        privatefield.y = (double)privatespin.y * (invDistCube - 3 * invDistPenta * r.y * r.y) - (double)privatespin.z * 3 * invDistPenta * r.y * r.z - (double)privatespin.x * 3 * invDistPenta * r.y * r.x; 
        privatefield.z = (double)privatespin.z * (invDistCube - 3 * invDistPenta * r.z * r.z) - (double)privatespin.x * 3 * invDistPenta * r.z * r.x - (double)privatespin.y * 3 * invDistPenta * r.z * r.y; 

        

        if (distSqr < tresh) {

            float J = Jxy;

            //if (privatepos.w > 0.0f && nextPos.w > 0.0f) {
            //    J = Jyy;
            //} else if (privatepos.w < 0.0f && nextPos.w < 0.0f) {
            //    J = Jxx;
            //}                

            privatefield.x = privatefield.x + (double)(privatespin.x * J);
            privatefield.y = privatefield.y + (double)(privatespin.y * J);
            privatefield.z = privatefield.z + (double)(privatespin.z * J);
        }

        localSums[tid] = privatefield;
    }
	
	} else {
	
		localSums[tid].x = 0;
        localSums[tid].y = 0;
        localSums[tid].z = 0;
	}

    // Loop for computing localSums : divide WorkGroup into 2 parts
    for (unsigned int stride = localSize/2; stride>0; stride /=2)
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

    if (gid == 0) {
        partialField[0].w = pos[next].w;
    }    

}