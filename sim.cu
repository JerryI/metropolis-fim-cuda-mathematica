#include <fstream>
#include <iostream>
#include <blaze/Blaze.h>
#include <blaze/Forward.h>

//nvcc -O2 -Xptxas -O2,-v sim.cu -llapack -lblas -o sim && ./sim tests/15x15x2-dcubic-float-reduceupdateGPUnew 4096 4 0.0444444 0.0 -10.3 -0.444444 1.21 4 0.1 10 300 300 15 20.0 50

#include <chrono>
#include <random>

using blaze::unaligned;
using blaze::unpadded;
using blaze::rowMajor;
using blaze::columnMajor;

#include <limits.h>

#define THREADS 1024

std::string dirname;
int NS;
int PARTIAL_SIZE;

using T = float4;

float dip, jxx, jyy, jxy, tresh, hitemp, lotemp;

int cycles, width, height, df;
float att;

long max_allowed_cycles = 1000;

#define STATE_SIZE 4
#define PROP_SIZE 4

#define SAMPLES 1000000

#include "includes/bmpdrawer.h"

#include "includes/cudaf.h"
#include "includes/gpu_reduce.hpp"


blaze::HermitianMatrix< blaze::StaticMatrix<std::complex<float>,10UL, 10UL> > constA;
blaze::HermitianMatrix< blaze::StaticMatrix<std::complex<float>,10UL, 10UL> > SxA;
blaze::HermitianMatrix< blaze::StaticMatrix<std::complex<float>,10UL, 10UL> > SyA;
blaze::HermitianMatrix< blaze::StaticMatrix<std::complex<float>,10UL, 10UL> > SzA;

float* spino;
float* poso;

void readfiles(std::string path) {
    std::ifstream file_constA( path + "/static_a.bin", std::ios::binary );
    std::ifstream file_sxA( path + "/sx_a.bin", std::ios::binary );
    std::ifstream file_syA( path + "/sy_a.bin", std::ios::binary );
    std::ifstream file_szA( path + "/sz_a.bin", std::ios::binary );

    std::ifstream file_spins( path + "/init.bin", std::ios::binary );
    std::ifstream file_poss(  path + "/coordinates.bin", std::ios::binary );

    file_constA.read((char*)constA.data(), 10*10*2*sizeof(float));
    file_sxA.read((char*)SxA.data(), 10*10*2*sizeof(float));
    file_syA.read((char*)SyA.data(), 10*10*2*sizeof(float));
    file_szA.read((char*)SzA.data(), 10*10*2*sizeof(float));

    spino = new float[NS*STATE_SIZE];
    poso = new float[NS*PROP_SIZE];

    file_spins.read((char*)spino, NS*STATE_SIZE*sizeof(float));
    file_poss.read((char*)poso, NS*PROP_SIZE*sizeof(float));
}

void makesnapshot(std::string path, int c) {
    std::string sfilename = path + "/snapshots/cycle_" + std::to_string(c)  + ".bin";
    std::ofstream file_snap(sfilename, std::ios::trunc);
    file_snap.write((char*)spino, sizeof(float) * NS * STATE_SIZE);
    file_snap.close();
}

inline void cuCheck(cudaError_t code){
    if(code != cudaSuccess){
        std::cerr << "Error code: " << code << std::endl << cudaGetErrorString(code) << std::endl;
    }
}

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;    
    //std::cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << std::endl << std::endl; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " << std::endl << std::endl;

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
        std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
        std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        std::cout << "  Warp size:         " << props.warpSize << std::endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::cout << std::endl;
    }
}

std::random_device dev;
std::mt19937 rng(dev());

std::uniform_real_distribution<float> dis(0.0, 1.0);

double sum_of_weight = 0;
double rnd = 0;

int randomChoice(double *choice_weight, int num_choices) {
    sum_of_weight = 0.0;
    for(int i=0; i<num_choices; i++) {
        sum_of_weight += choice_weight[i];
    }
    rnd = ((double)dis(rng))*sum_of_weight;
    //rnd = sum_of_weight*(double)rand()/(double)(RAND_MAX);

    for(int i=0; i<num_choices; i++) {
        if(rnd < choice_weight[i])
            return i;
        rnd -= choice_weight[i];
    }

    std::cout  << "ERROR";
    std::cout << "\n" << "variants:\n";
    for(int i=0; i<num_choices; ++i) {
        std::cout << choice_weight[i] << "\n";
    }
    std::exit(-1);
}

float4* d_spin, *d_pos, *d_partial;

float temperature; 
int nextSpin = 1;
int prevSpin = 1;
float invTemp;

void CUDAinit() {
    cudaMalloc((void**)&d_spin,    NS*STATE_SIZE*sizeof(float));
    cudaMalloc((void**)&d_pos,     NS*PROP_SIZE*sizeof(float));
    cudaMalloc((void**)&d_partial, 4*PARTIAL_SIZE*sizeof(float));

    cudaMemcpy((void*)d_spin, (void*)spino, sizeof(float) * NS * STATE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_pos, (void*)poso, sizeof(float) * NS * PROP_SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void CUDAterm() {

}

void testRng() {
    std::cout << "spin next: " << spino[nextSpin*4] << "; " << spino[nextSpin*4 + 1] << "; " << spino[nextSpin*4 + 2] << "\n";
    std::cout << "type next: " << poso[nextSpin*4 + 3]  << "\n";
    std::cout << "spin prev: " << spino[prevSpin*4] << "; " << spino[prevSpin*4 + 1] << "; " << spino[prevSpin*4 + 2] << "\n";
    std::cout << "type prev: " << poso[prevSpin*4 + 3]  << "\n";    
}

void testNeib() {
    std::cout << "counting neibours\n";

    int *atlas = new int[28];
    for (int i =0; i<28; ++i) {
        atlas[i] = 0;
    }

    float4 delta;
    float snorm;
    int cnt;
    float4 *initpos = (float4*)poso;
    float4 *ppos = (float4*)poso; 
    for (int j=0; j<NS; ++j) {
        cnt=0;
        ppos = (float4*)poso; 
        for (int i=0; i<NS; ++i) {
            delta = (*ppos) - (*initpos);
            snorm = (delta.x*delta.x + delta.y*delta.y +delta.z*delta.z);
            if (snorm < tresh && snorm > 0.1f) ++cnt;
            ++ppos;
        }

        ++atlas[cnt];
        ++initpos;
    }

    int cntmax = 0;
    for (int i=0; i<28; ++i)
        if (atlas[i] > cntmax) cntmax = atlas[i];

    std::cout << "------hystogram------\n";
    for (int i=0; i<28; ++i) {
        std::cout << " " << i << "\t";
        for (int level=0; level<(int)(std::floor( 100.0f*((float)atlas[i])/((float)cntmax) )); ++level) {
            std::cout << ":";
        }
        std::cout << "\n";
    }

    free(atlas);
}

void testEigen() {
    std::cout << "eigen test\n <Sx>\n";
    
    blaze::HermitianMatrix< blaze::StaticMatrix<std::complex<float>,10UL, 10UL> > hamiltonianA = SxA + SyA + SzA;
    blaze::DynamicVector<float,blaze::columnVector>      wA (10UL);       // The vector for the real eigenvalues
    blaze::DynamicMatrix<std::complex<float>,blaze::rowMajor> VA (10UL, 10UL);  // The matrix for the left eigenvectors
 
    blaze::eigen(hamiltonianA, wA, VA ); 
        
    for(int i=0; i<10; ++i)
        std::cout << wA[i] << "\t";
    std::cout << "\n";

    for (int j=0; j<10; ++j) {
        std::cout << "vector: " << j << "\n";
        for(int i=0; i<10; ++i)
            std::cout << row( VA, j )[i].real() << " + i*" << row( VA, j )[i].imag() << "\t\t";
        std::cout << "\n";
    }

    std::cout << "\nthe probe state will be\n";
    for(int i=0; i<10; ++i) {
        auto probe = conj(row( VA,i) );

        std::cout << (conj(probe) * (SxA * trans(probe) )).real() << "\t";
        std::cout << (conj(probe) * (SyA * trans(probe) )).real() << "\t";
        std::cout << (conj(probe) * (SzA * trans(probe) )).real() << "\t\n";
    }

    std::cout << "\n";
}

int samples[SAMPLES];

//blaze::StaticVector<float, 4UL> newSpin;
blaze::StaticVector<float, 4UL> energiesB = {3.0/2.0, 1.0/2.0, -1.0/2.0, -3.0/2.0};

blaze::HermitianMatrix< blaze::StaticMatrix<std::complex<float>,10UL, 10UL> > hamiltonianA;
blaze::DynamicVector<float,blaze::columnVector>      wA(10UL);       // The vector for the real eigenvalues
blaze::DynamicMatrix<std::complex<float>,blaze::rowMajor> VA(10UL, 10UL);  // The matrix for the left eigenvectors
blaze::StaticVector<double, 10UL> boltzmanA;

auto st = row(VA, 0);

inline void calcFe(blaze::DynamicVector<float, blaze::rowVector> total, float4* spin) {
    blaze::eigen(constA + SyA*total[1] + SxA*total[0] +  SzA*total[2], wA, VA );
    boltzmanA = - wA * invTemp;
    boltzmanA = exp(boltzmanA);

    st = row( VA, randomChoice((double*)boltzmanA.data(), 10));

    spin->x = (st * (SxA * ctrans(st) )).real();
    spin->y = (st * (SyA * ctrans(st) )).real();
    spin->z = (st * (SzA * ctrans(st) )).real();
}

//blaze::StaticVector<float, 3UL> sv1;
//blaze::StaticVector<double, 4UL> boltzmanB;
//float normsv;
blaze::StaticVector<float, 4UL> tot;

inline void calcCr(float4* spin) {
    //std::cout << tot[0] << "; " << tot[1] << "; " << tot[2] << "\n";

    auto normsv = sqrtf(tot[0]*tot[0]+tot[1]*tot[1]+tot[2]*tot[2]);

    //std::cout << normsv << "\n";

    blaze::StaticVector<double, 4UL> boltzman = - energiesB * normsv * invTemp;
    boltzman = exp(boltzman);

    tot = tot / normsv;

    normsv = energiesB[randomChoice((double*)boltzman.data(), 4)];

    tot = tot * normsv;

    spin->x = tot[0];
    spin->y = tot[1];
    spin->z = tot[2];    
}

void round() {
    float4 newSpin_storage;

    newSpin_storage.x = 0.0;
    newSpin_storage.y = 0.0;
    newSpin_storage.z = 0.0;

    size_t n = NS;
    size_t blocksPerGrid = std::ceil((1.*n) / THREADS);

    T* tmp;
    T* from;

    cudaMalloc(&tmp, sizeof(T) * blocksPerGrid); 
    checkCUDAError("Error allocating tmp [GPUReduction]");

    float4* buf = new float4[blocksPerGrid];

    for (int k=0; k<SAMPLES; ++k) {
        nextSpin = samples[k];

        n = NS;
        blocksPerGrid   = std::ceil((1.*n) / THREADS);

        /*
            std::cout << "number: " << n << "\n";
            std::cout << "blokcs: " << blocksPerGrid << "\n";
        */


        reduceCUDAPopulate<THREADS><<<blocksPerGrid, THREADS>>>(d_spin, d_pos, newSpin_storage, jxx, jyy, jxy, dip, tresh, tmp, nextSpin, prevSpin, n);
        from = tmp;
        n = blocksPerGrid;

        //cudaDeviceSynchronize();

        #ifdef many_parts
        do
        {
            blocksPerGrid   = std::ceil((1.*n) / THREADS);
            reduceCUDA<THREADS><<<blocksPerGrid, THREADS>>>(from, tmp, n);
            from = tmp;
            n = blocksPerGrid;
        } while (n > THREADS);

        if (n > 1)
            reduceCUDA<THREADS><<<1, THREADS>>>(tmp, tmp, n);  

        cudaMemcpy(tot.data(), tmp, sizeof(float)*3, cudaMemcpyDeviceToHost);       
        #endif

        //___works faster it we use CPU to calculate the rest of the sum___


        //CPU Total
        cudaMemcpy(buf, tmp, sizeof(float)*4*n, cudaMemcpyDeviceToHost);

        *((float4*)tot.data()) = buf[0];

        for (int i=1; i<n; ++i) {
            *((float4*)tot.data()) += buf[i];
        } 
        //checkCUDAError("Error copying result [GPUReduction]");
    
        calcCr(&newSpin_storage);
        prevSpin = samples[k];
    
    }

    free(buf);
    cudaFree(tmp);
}

void calcMagnetization() {
    blaze::DynamicMatrix<float> storage(PARTIAL_SIZE, 4);

    float* storage_s = (float*)storage.data();

    calcMag<<<PARTIAL_SIZE, THREADS>>>((float4*)d_spin, (float4*)d_partial, NS);
    cudaMemcpy((void*)storage_s, (void*)d_partial, sizeof(float) * 4 * PARTIAL_SIZE, cudaMemcpyDeviceToHost);
    
    blaze::DynamicVector<float, blaze::rowVector> mag = blaze::sum<blaze::columnwise>(storage);
    std::cout << "Magnetization: " << 3.0f*mag[0]/((float)NS) << "; " << 3.0f*mag[1]/((float)NS) << "; " << 3.0f*mag[2]/((float)NS) << "\n";
}

void endless() {
    DisplayHeader();

    CUDAinit();
    testEigen();

    #ifdef DOUBLEP
        std::cout << "DOUBLE mode is enabled!!! \n";
    #endif

    std::cout << "-----------------parameters-----------------" << "\n";
    std::cout << "Jxx = " << jxx << "; Jyy = " << jyy << "; Jxy = " << jxy << "; D = " << dip << "\n";
    std::cout << "--------------------------------------------" << "\n\n";

    std::uniform_int_distribution<std::mt19937::result_type> random_sampler(0, NS-1);


    std::chrono::steady_clock::time_point beginGlobal = std::chrono::steady_clock::now();

    for(int cycle = 0; cycle < cycles; ++cycle) {
        temperature = lotemp + ((float)(cycles - cycle)/((float)cycles))*(hitemp-lotemp);
        invTemp = 1.0f/(temperature * 0.695f); 

        std::cout << "Temperature: " << temperature << "\n";
        std::cout << "Cycle: " << cycle << "\n";

        std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();

        for (int i=0; i<SAMPLES; ++i) samples[i] = random_sampler(rng);

        round();

        std::chrono::steady_clock::time_point endtime = std::chrono::steady_clock::now();

        std::cout << "time elapsed  = " << std::chrono::duration_cast<std::chrono::seconds>(endtime - time).count() << "[s]" << std::endl;
        std::cout << "speed  = " << ((float)SAMPLES)/((float)(std::chrono::duration_cast<std::chrono::seconds>(endtime - time).count())) << "[sam/sec]" << std::endl;
        std::cout << "waiting time  = " << ((float)(cycles - cycle)*(std::chrono::duration_cast<std::chrono::seconds>(endtime - time).count()))/(60.0f*60.0f) << "[hrs]" << std::endl;

        std::chrono::steady_clock::time_point extratime = std::chrono::steady_clock::now();

        cudaMemcpy(spino, d_spin, sizeof(float) * NS * STATE_SIZE, cudaMemcpyDeviceToHost);
        makesnapshot(dirname, cycle);
        drawpicture((dirname + "/snapshots/cycle_" + std::to_string(cycle) + ".bmp").c_str(), width, height, df, att, poso, spino);
        calcMagnetization();

        std::chrono::steady_clock::time_point extratimeend = std::chrono::steady_clock::now();
        std::cout << "post time elapsed  = " << std::chrono::duration_cast<std::chrono::seconds>(extratimeend - extratime).count() << "[s]" << std::endl;
        std::cout << "\n\n";
    

    }

    std::chrono::steady_clock::time_point endGlobal = std::chrono::steady_clock::now();
    std::cout << "\n\n";
    std::cout << "total time  = " << std::chrono::duration_cast<std::chrono::seconds>(endGlobal - beginGlobal).count() << "[s]" << std::endl;
    std::cout << "total time  = " << ((float)std::chrono::duration_cast<std::chrono::seconds>(endGlobal - beginGlobal).count())/(60.0f*60.0f) << "[hr]" << std::endl;

    std::cout << "The rest will be endless... \n";

    long cycle = cycles;

    while(true) {
        for (int i=0; i<SAMPLES; ++i) samples[i] = random_sampler(rng);
        round();
        cudaMemcpy(spino, d_spin, sizeof(float) * NS * STATE_SIZE, cudaMemcpyDeviceToHost);
        makesnapshot(dirname, cycle);
        drawpicture((dirname + "/snapshots/cycle_" + std::to_string(cycle) + ".bmp").c_str(), width, height, df, att, poso, spino);
        calcMagnetization();   
        cycle++;     

        if (cycle > max_allowed_cycles) break;
    }        

    std::cout << "terminated.";

    //neve be reached

}


int main(int argc, char** argv)
{
    

    dirname = argv[1];

    NS =    std::stoul(argv[2]);
    PARTIAL_SIZE =    std::stoi(argv[3]);

    dip =   std::stof(argv[4]);
    jxx =   std::stof(argv[5]);
    jyy =   std::stof(argv[6]);
    jxy =   std::stof(argv[7]);
    tresh =   std::stof(argv[8]);
    hitemp =   std::stof(argv[9]);
    lotemp =   std::stof(argv[10]);

    cycles =   std::stoi(argv[11]);
    width  =   std::stoi(argv[12]);
    height =   std::stoi(argv[13]);
    df     =   std::stoi(argv[14]);

    att    =   std::stof(argv[15]);
    max_allowed_cycles = std::stoi(argv[16]);

    readfiles(dirname);

    endless();

    return 0;
}