#include "spline.h"
#include "solve_system.h"
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define cuAlloc(x, y) cudaMalloc((void**)&(x), (y))
#define copyToCard(x, y, z) cudaMemcpy((x), (y), (z), cudaMemcpyHostToDevice);
#define copyToHost(x, y, z) cudaMemcpy((x), (y), (z), cudaMemcpyDeviceToHost);

static Point **m_generatePoints(Point **, int, int, int);
__global__ void fillA(float *, Point *);
__global__ void fillb(float *, Point *);
__global__ void solveSplineCu(CubicCurve *, Point*);

/**
 * Checks that numSets, numPoints, and granularity are all above zero.
 * If granularity is 1, we just return the start points unchanged.
 * Otherwise, we generate points.
 */
Point** generatePoints(Point ** start, int numSets, int numPoints, int gran) {
   assert(numSets > 0);
   assert(numPoints > 0);
   assert(gran > 0);

   if (gran == 1) {
      return start;
   }
   else {
      return m_generatePoints(start, numSets, numPoints, gran);
   }
}

static Point **m_generatePoints(Point **start, int numSets, int numPoints, int gran) {
   static int sets = 1, granularity = 1, points = 2;
   static Point** ret = (Point **) malloc(sets * sizeof(Point *));
   static Point* ans = (Point*) malloc((points - 1) * sets * granularity * sizeof(Point));

   CubicCurve **ss;
   int set, totFinalPoints = (numPoints - 1) * numSets * gran;
   CubicCurve *ss_d;
   Point *ans_d;
   
   if (numSets != sets) {
      sets = numSets;
      free(ret);
      ret = (Point **) malloc(numSets * sizeof(Point *));
   }

   if (numSets != sets || gran != granularity || numPoints != points) {
      sets = numSets;
      granularity = gran;
      points = numPoints;
      free(ans);
      ans = (Point*) malloc(totFinalPoints * sizeof(Point));
   }

   ss = generateSplines(start, numSets, numPoints);

   cudaDeviceSynchronize();

   cuAlloc(ss_d, (numPoints - 1) * numSets * sizeof(CubicCurve));
   cuAlloc(ans_d, totFinalPoints * sizeof(Point));

   for (set = 0; set < numSets; set++) {
      copyToCard(&(ss_d[set * (numPoints - 1)]), ss[set], (numPoints - 1)
       * sizeof(CubicCurve));
   }

   for (set = 0; set < numSets; set++) {
      solveSplineCu<<<(numPoints - 1), gran>>>
       (&(ss_d[set * (numPoints - 1)]), &(ans_d[set * (numPoints - 1) * gran]));
   }

   copyToHost(ans, ans_d, totFinalPoints * sizeof(Point));

   for (set = 0; set < numSets; set++) {
      ret[set] = &(ans[set * (numPoints - 1) * gran]);
   }

   cudaFree(ans_d);
   cudaFree(ss_d);

   for (set = 0; set < numSets; set++)
      free(ss[set]);

   free(ss);

   return ret;
}

/**
 * Generates a list of splines from the list of points pts. The number of
 * elements in pts should equal num. The number of splines returned is then
 * num - 1.
 */
CubicCurve **generateSplines(Point **pts, int splines, int num) {
   CubicCurve **ss = (CubicCurve **) malloc(splines * sizeof(CubicCurve *));

   int i, j, rows = num - 2, elems = (rows * rows);
   float *A_d, *b_d; 
   float *A = (float *) calloc(elems, sizeof(float));
   Point *pts_d;

   // Malloc on the card: Ax = b
   cudaMalloc((void**)&A_d, elems * sizeof(float));
   cudaMalloc((void**)&b_d, splines * rows * sizeof(float));
   cudaMalloc((void**)&pts_d, splines * num * sizeof(Point));
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess)
      printf("spline: error mallocing on device.\n");

   // Copies A to A_d: ensures the CUDA buffer is initially set to 0's 
   cudaMemcpy(A_d, A, elems * sizeof(float), cudaMemcpyHostToDevice);
   free(A);

   for (i = 0; i < splines; i++) {
      // Copy point data to card.
      cudaMemcpy(&(pts_d[i * num]), pts[i], num * sizeof(Point), cudaMemcpyHostToDevice);
   }

   error = cudaGetLastError();
   if (error != cudaSuccess)
      printf("spline: error mallocing on device.\n");

   fillA<<<1, rows>>>(A_d, pts_d);

   error = cudaGetLastError();
   if (error != cudaSuccess)
      printf("spline: error mallocing on device.\n");

   // Perform computation to fill A_d and b_d. 
   // A: coefficients of z-values in equations. Each row of matrix is one eqn.
   // b: right-hand-side of the equations.
   // (Ax = b), where A is a matrix, and x and b are column vectors.
   fillb<<<splines, rows>>>(b_d, pts_d);
   
   error = cudaGetLastError();
   if (error != cudaSuccess)
      printf("spline: error mallocing on device.\n");

   // Make sure fillMatrices is done.
   cudaDeviceSynchronize();
   cudaFree(pts_d);

   float *x = solveSystem(A_d, b_d, splines, rows);

   for (j = 0; j < splines; j++) {
      // Now that we have the matrix, solve for x. X then holds our Z-values
      // (the second derivative at the points).
      CubicCurve *cc = (CubicCurve *) malloc((num - 1) * sizeof(CubicCurve));

      cc[0].z1 = 0;
      cc[num - 2].z2 = 0;

      // Fill CubicCurve array with Z-values and points
      for (i = 0; i < num - 1; i++) {
         if (i != 0)
            cc[i].z1 = x[j * rows + (i - 1)];

         cc[i].p1 = pts[j][i];
         cc[i].p2 = pts[j][i + 1];

         if (i != num - 2)
            cc[i].z2 = x[j * rows + i];
      }

      ss[j] = cc;
   }

   cudaFree(A_d);
   cudaFree(b_d);

   free(x);

   return ss;
}

__shared__ float h[TILE_SIZE];

/**
 * Fills matrix A with coefficients of equations (where Z-values are vars)
 * and fills vector b with the right-hand-sides of the linear equation.
 * The length of pts should be 2 more than blockDim.x.
 * GridDim.x should be the number of matrices to fill.
 */
__global__ void fillA(float *A, Point *pts) {
   int j = 0;
   int pos = threadIdx.x;
   int rows = blockDim.x;

   // Initialize h values: h-sub-i is the distance between point i+1 and point
   // i on the x-axis.
   if (pos == rows - 1) h[pos + 1] = pts[pos + 2].x - pts[pos + 1].x;
   h[pos] = pts[pos + 1].x - pts[pos].x;   
   __syncthreads();

   j = (pos % rows == 0) ? 0 : pos - 1;

   // If we're not on row 0, fill the element left of the diagonal with h[pos]
   if (pos != 0)
      A[pos * rows + j++] = h[pos];

   // For every row, fill the diagonal element
   A[pos * rows + j++] = 2*(h[pos + 1] + h[pos]);

   // If we're not on the last row, fill the element right of the diag.
   if (pos != rows - 1)
      A[pos * rows + j++] = h[pos + 1];
}

__global__ void fillb(float *b, Point *pts) {
   int pos = blockIdx.x * blockDim.x + threadIdx.x, x = threadIdx.x;
   int ptpos = blockIdx.x * (blockDim.x + 2) + threadIdx.x;
   int rows = blockDim.x;

   // Initialize h values: h-sub-i is the distance between point i+1 and point
   // i on the x-axis.
   if (x == rows - 1) h[x + 1] = pts[ptpos + 2].x - pts[ptpos + 1].x;
   h[x] = pts[ptpos + 1].x - pts[ptpos].x;   
   __syncthreads();

   // Fill b with this formula. 
   if (x < rows)
      b[pos] = 6*((pts[ptpos + 2].y - pts[ptpos + 1].y) / h[x + 1]
       - (pts[ptpos + 1].y - pts[ptpos].y) / h[x]);
}
/**
 * Takes the cubic spline described by cc and solves it.
 * blockDim.x indicates the granularity of the answer per curve.
 * gridDim.x indicates the number of curves to solve.
 *
 * Makes blockDim.x - 1 points between each two sets of curves.
 */
__global__ void solveSplineCu(CubicCurve *cc, Point *ans) {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   int cc_num = blockIdx.x;
   int x_num = threadIdx.x;

   CubicCurve cv = cc[cc_num];
   float h = cv.p2.x - cv.p1.x;

   float x = cv.p1.x + (h / blockDim.x) * ((float) x_num);

   float fx1 = x - cv.p1.x;
   float fx2 = cv.p2.x - x;

   ans[idx].x = x;
   ans[idx].y = (cv.z2 / (6.0*h)) * (fx1 * fx1 * fx1)
    + (cv.z1 / (6.0*h)) * (fx2 * fx2 * fx2)
    + (cv.p2.y / h - (h / 6.0) * cv.z2) * fx1
    + (cv.p1.y / h - (h / 6.0) * cv.z1) * fx2;
}
