# metropolis-fim-cuda-mathematica

Type
```shell
wolframscript -file generate.wls 7x7x2 df=10 cycles=50 hitemp=100 att=30.0
```
it will generate and evolve spin cluster 7x7x2 of $Fe Cr_2 S_4$ cells. It contains two magnetic subblatice with a strong cubic anisotropy. Each spin is threated as a real quantum system including true orbital degrees of freedom.

## Required 
- Freeware Wolfram Engine
- CUDA
- blaze-lib
- blas/lapack
- C++14

## Compilation
```shell
nvcc -O2 -Xptxas -O2,-v sim.cu -llapack -lblas -o sim
```
