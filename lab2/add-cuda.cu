//
// Created by vad on 16/10/21.
// DI FCTUNL
//

#include <assert.h>
#include <cuda.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>

#include "vsize.h"

// intVec - initialize all v with val
void initVec(float *v, float val) {
  for (int i = 0; i < VSIZE; i++)
    v[i] = val;
}

// checkVec - check that all v has val
void checkVec(float *v, float val) {
  for (int i = 0; i < VSIZE; i += 1000) {
    // printf("[%d] -> %f\n", i, v[i]);
    assert(v[i] == val);
  }
}

/* SAXPY BLAS like function
 * single precision X=a*X+Y  (X and Y vectors)
 * out: result in X
 */
__global__ void saxpy(float a, float *X, float *Y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  i = i * 16;
  // TODO:
  if (i <= VSIZE) {
    for (int j = 0; j < 16; j++)
      X[i + j] = a * X[i + j] + Y[i + j];
  }
}

/******* MAIN ******/

int main(int argc, char *argv[]) {
  size_t blockSize = GROUPSIZE;
  if (argc == 2)
    blockSize = atoi(argv[1]);
  else if (argc > 2) {
    fprintf(stderr, "usage: %s [block_size]\n", argv[0]);
    return EXIT_FAILURE;
  }
  printf("starting...\n");
  int v = (VSIZE + 15) / 16;
  size_t numBlocks = (v + blockSize - 1) / blockSize;
  printf("Using %zu threads, %u blocks of %u threads\n", blockSize * numBlocks,
         numBlocks, blockSize);

  float *a = (float *)malloc(VSIZE * sizeof(float));
  float *b = (float *)malloc(VSIZE * sizeof(float));
  if (a == NULL || b == NULL) {
    fprintf(stderr, "No mem!\n");
    return EXIT_FAILURE;
  }
  initVec(a, 1.0);
  initVec(b, 2.0);

  float *d_a;
  cudaMalloc(&d_a, VSIZE * sizeof(float));
  float *d_b;
  cudaMalloc(&d_b, VSIZE * sizeof(float));
  if (d_a == NULL || d_b == NULL) {
    fprintf(stderr, "No GPU mem!\n");
    return EXIT_FAILURE;
  }
  cudaMemcpy(d_a, a, VSIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, VSIZE * sizeof(float), cudaMemcpyHostToDevice);

  clock_t t = clock();
  saxpy<<<numBlocks, blockSize>>>(2.0, d_a, d_b);
  t = clock() - t;
  printf("time: %f ms\n", t / (double)CLOCKS_PER_SEC * 1000.0);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "err=%u %s\n%s\n", (unsigned)err, cudaGetErrorString(err),
            "Problems executing kernel");
    exit(1);
  }

  cudaMemcpy(a, d_a, VSIZE * sizeof(float), cudaMemcpyDeviceToHost);

  checkVec(a, 4.0); // 2*1+2 == 4
  printf("OK\n");

  cudaFree(d_a);
  cudaFree(d_b);

  return EXIT_SUCCESS;
}
