#ifndef SOLVE_SYSTEM_H
#define SOLVE_SYSTEM_H

#define TILE_SIZE 400
#define MAT_SIZE 400
#define MAXERR 1e-2
#define MAX_GRID_SIZE 65535

#define checkCudaError(x) {\
   error = cudaGetLastError();\
   if (error != cudaSuccess) {\
      printError(x);\
      exit(-1);\
   }\
}

#define printError(x) printf(x " #%d, %s\n", error, cudaGetErrorString(error))

float *solveSystem(float *, float *, int, int);

#endif
