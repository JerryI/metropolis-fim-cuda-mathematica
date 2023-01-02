#include <fstream>
#include <iostream>
#include <blaze/Blaze.h>
#include <blaze/Forward.h>

#include <chrono>
#include <random>

using blaze::unaligned;
using blaze::unpadded;
using blaze::rowMajor;
using blaze::columnMajor;

std::string dirname;
unsigned long NS;
int PARTIAL_SIZE;

float dip, jxx, jyy, jxy, tresh, hitemp, lotemp;

int cycles, width, height, df;
float att;

long max_allowed_cycles = 1000;

#define STATE_SIZE 4
#define PROP_SIZE 4

#define SAMPLES 1000000

#include "includes/bmpdrawer.h"
#include "includes/cuda.h"

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

float* d_spin, *d_pos, *d_partial;

float temperature; 
unsigned long nextSpin = 1;
unsigned long prevSpin = 1;
float invTemp;

void CUDAinit() {
    cudaMalloc((void**)&d_spin,    NS*STATE_SIZE*sizeof(float));
    cudaMalloc((void**)&d_pos,     NS*PROP_SIZE*sizeof(float));
    cudaMalloc((void**)&d_partial, 4*PARTIAL_SIZE*sizeof(float));

    cudaMemcpy((void*)d_spin, (void*)spino, sizeof(float) * NS * STATE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_pos, (void*)poso, sizeof(float) * NS * PROP_SIZE, cudaMemcpyHostToDevice);
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

    unsigned long *atlas = new unsigned long[28];
    for (int i =0; i<28; ++i) {
        atlas[i] = 0;
    }

    float4 delta;
    float snorm;
    unsigned long cnt;
    float4 *initpos = (float4*)poso;
    float4 *ppos = (float4*)poso; 
    for (unsigned long j=0; j<NS; ++j) {
        cnt=0;
        ppos = (float4*)poso; 
        for (unsigned long i=0; i<NS; ++i) {
            delta = (*ppos) - (*initpos);
            snorm = (delta.x*delta.x + delta.y*delta.y +delta.z*delta.z);
            if (snorm < tresh && snorm > 0.1f) ++cnt;
            ++ppos;
        }

        ++atlas[cnt];
        ++initpos;
    }

    unsigned long cntmax = 0;
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

unsigned long samples[SAMPLES];

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

blaze::StaticVector<float, 3UL> sv1;
blaze::StaticVector<double, 4UL> boltzmanB;
double normsv;

inline void calcCr(blaze::DynamicVector<float, blaze::rowVector> total, float4* spin) {
    sv1[0] = total[0];
    sv1[1] = total[1];
    sv1[2] = total[2]; 

    normsv = sqrtf(sv1[0]*sv1[0]+sv1[1]*sv1[1]+sv1[2]*sv1[2]);
    sv1 = sv1 / normsv;

    boltzmanB = - energiesB * normsv * invTemp;
    boltzmanB = exp(boltzmanB);

    normsv = energiesB[randomChoice((double*)boltzmanB.data(), 4)];

    spin->x = sv1[0]*normsv;
    spin->y = sv1[1]*normsv;
    spin->z = sv1[2]*normsv;    
}

void round() {
    //accumulator for the calcualted fields
    blaze::DynamicMatrix<float> partialField(PARTIAL_SIZE, 4);
    float* partialField_storage = (float*)partialField.data();

    float4 newSpin_storage;

    newSpin_storage.x = 0.0;
    newSpin_storage.y = 0.0;
    newSpin_storage.z = 0.0;

    

    for (unsigned long k=0; k<SAMPLES; ++k) {
        nextSpin = samples[k];

        calcFields<<<PARTIAL_SIZE, 1024>>>((float4*)d_spin, (float4*)d_pos, newSpin_storage, jxx, jyy, jxy, dip, tresh, (float4*)d_partial, nextSpin, prevSpin, NS);
        cudaMemcpy((void*)partialField_storage, (void*)d_partial, sizeof(float) * 4 * PARTIAL_SIZE, cudaMemcpyDeviceToHost);

        //for(int i=0; i<PARTIAL_SIZE; ++i) {
        //    std::cout << "v1: " << partialField_storage[4*i] << ", v2: " << partialField_storage[4*i+1] << ", v3: " << partialField_storage[4*i+2] << ", v4: " << partialField_storage[4*i+3] << "\n";
        //}

     

        if (partialField_storage[3] < 0.0f)
            calcFe(blaze::sum<blaze::columnwise>(partialField), &newSpin_storage);
        else
            calcCr(blaze::sum<blaze::columnwise>(partialField), &newSpin_storage);

        prevSpin = samples[k];
     
    }

   
    


}

void calcMagnetization() {
    blaze::DynamicMatrix<float> storage(PARTIAL_SIZE, 4);

    float* storage_s = (float*)storage.data();

    calcMag<<<PARTIAL_SIZE, 1024>>>((float4*)d_spin, (float4*)d_partial, NS);
    cudaMemcpy((void*)storage_s, (void*)d_partial, sizeof(float) * 4 * PARTIAL_SIZE, cudaMemcpyDeviceToHost);
    
    blaze::DynamicVector<float, blaze::rowVector> mag = blaze::sum<blaze::columnwise>(storage);
    std::cout << "Magnetization: " << 3.0f*mag[0]/((float)NS) << "; " << 3.0f*mag[1]/((float)NS) << "; " << 3.0f*mag[2]/((float)NS) << "\n";
}

void endless() {
    CUDAinit();
    testEigen();

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