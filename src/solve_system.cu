#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "solve_system.h"

#define EPSILON 1e-2
#define ITERS 50

__global__ void jacobiMethod(float* coal, float* x, float* b, int col, int num);

/**
 * Solves the system pointed to by _device_ pointers A_d and b_d, and non-
 * device pointer x. Col must be the number of elements in x and b_d, and
 * col^2 is the number of elements in A_d.
 */
float* solveSystem(float* A_d, float* b_d, int splines, int col) {
   float *x_d, *x = (float *) calloc(splines * col, sizeof(float));
   int numA = col * col;
   cudaError_t error;

   // Malloc space on the card.
   cudaMalloc((void**)&x_d, splines * col * sizeof(float));
   checkCudaError("solve_system.cu: error mallocing on device:");

   // Copy the initial guess x onto the card.
   cudaMemcpy(x_d, x, splines * col * sizeof(float), cudaMemcpyHostToDevice);
   checkCudaError("solve_system.cu: error mem copying to device:");

   // Run all iters of the jacobi method.
   jacobiMethod<<<splines, col>>>(A_d, x_d, b_d, col, numA);
   checkCudaError("solve_system.cu: error calling CUDA kernel:");

   free(x);

   return x_d;
}
   
__shared__ float psum[MAT_SIZE];

/**
 * Use the Jacobi Method (on wikipedia and others) to solve a linear system
 * of equations. This CUDA call executes one iteration of the method.
 * Multiple calls are expected to be made to find a suitable answer. Host
 * must copy x back and manually check whether convergence was reached.
 */ 
__global__ void jacobiMethod(float* coal, float* x, float* b, int col, int num) {
   int i;
   int idx = blockIdx.x * blockDim.x + threadIdx.x, d = 0;
   int elem = threadIdx.x * col;

   /*if (idx % blockDim.x < col)
      prev[elem] = 0;*/

   for (i = 0; i < ITERS; i++) {
   //do {
      // Set diagonal elements to zero; otherwise multiply by corresponding x val.
      for (d = 0; d < col; d++)
         if ((elem + d) < num && (elem + d) % (col + 1) != 0)
            psum[elem + d] = coal[elem + d] * x[blockIdx.x * blockDim.x + d];
         else
            psum[elem + d] = 0.0;

      __syncthreads();

      // Add up all elements in each line of the psum (shared mem) matrix
      for (d = 1; d < col; d++)
         psum[elem] += psum[(elem) + d];

      __syncthreads();

      // Compute the next guess for x using the psum array.
      if (threadIdx.x < col)
         x[idx] = (b[idx] - psum[elem]) / coal[threadIdx.x * (col + 1)];
   }
}
