// JacobiMethod.c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "JacobiMethod.h"

/* JacobiMethod() */

void jacobiMethod(float* coal, float* x, float* b, int col, int num) {
    int iteration, i , j;
    float error, sigma, biggestError;
    float temp[col];

    // Iterate NUMBER_OF_ITERATIONS times,or until convergence is reached
    for (iteration = 0; iteration < NUMBER_OF_ITERATIONS; iteration++) {
        biggestError = 0;  // stores the biggest error in the iteration

        // Generate Solution Matrix for iteration
        for (i = 0; i < col; i++) {
            sigma = 0;
            for (j = 0; j < col; j++) {
                if (j != i) {
                    sigma = sigma + (coal[(i * col) + j] * x[j]);
                }
            }
            //printf("sigma = %f\n", sigma);
            temp[i] = (b[i] - sigma)/coal[(i * col) + i];
            error = fabs((temp[i] - x[i])/temp[i]);

            // check to see if this was the largest error of the iteration
            if (error > biggestError)
                biggestError = error;
        }
        // move temp array to x array
        memcpy(x, temp, (sizeof(float) * col));

        printf("After %d iterations:\n", (iteration+1));
        int z;
        for (z = 0; z < col; z++) {
            printf("%f\n", x[z]);
        }
        printf("Largest Error = %f\n\n", biggestError);

        // Check for Convergence
        if (biggestError < ERROR_THRESHOLD) {
            printf("Threshold Reached\n");
            return;
        }
    }
}

int main() {
    float a[] = {
        5, -2, 3,
        -3, 9, 1,
        2, -1, -7
    };
    float x[] = {0,0,0};
    float b[] = {-1, 2, 3};

    jacobiMethod(a, x, b, 3, 9);

    /*float a[] = {
        2, 1,
        5, 7
    };
    float b[] = {11,13};
    float x[] = {1,1};

    jacobiMethod(a, x, b, 2, 4);*/

    return 0;
}
