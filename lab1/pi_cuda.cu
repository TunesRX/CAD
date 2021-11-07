#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>


#include <cuda.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS        256

#define NUM_STEPS2 (4096*256*256)


struct timeval ts, te;

void startTimer(){
    gettimeofday(&ts, NULL);
}

float stopTimer(){
  gettimeofday(&te, NULL);
  return (float) (te.tv_sec - ts.tv_sec)*1e6+
   (float) (te.tv_usec - ts.tv_usec);
}

/***/

// Cuda kernel
__global__ void piCalc( double *pi, long niter, double step, int threads, int blocks){
     double x;
     int idx = blockIdx.x*blockDim.x+threadIdx.x;
     int i;
     pi[idx] = 0.0;
     for(i = idx; i < niter; i+=threads*blocks){
	x = (double)i*step;
        pi[idx] += 1.0/(x*x+1.0);
     }
     pi[idx] = 4.0* (pi[idx])*step;
}

int main(int argc, char *argv[])
{
  double pi=0.0;
  float runtime;
  long num_steps2;

  if (argc==1) {
    num_steps2=NUM_STEPS2;
  } else if (argc==2) {
    num_steps2 = atoi(argv[1]);
  } else {
    fprintf(stderr, "usage: %s [num steps]\n", argv[0]);
    return 1;
  }
 
  printf("Started...\n");
  startTimer();
  double step = 1.0/num_steps2;
  int pi_size = NUM_BLOCKS*THREADS_PER_BLOCK*sizeof(double); //num bytes for parcial result
  double *pi_host = (double*)malloc(pi_size);
  if (pi_host==NULL) { fprintf(stderr, "No Mem!\n"); exit(1); }
  double *pi_dev;
  cudaMalloc(&pi_dev, pi_size);
  cudaMemset(pi_dev, 0, pi_size);


/* Then we calculate pi from integral 4/(1+x*x) */
  piCalc<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>( pi_dev, num_steps2, step, THREADS_PER_BLOCK, NUM_BLOCKS);

  cudaMemcpy(pi_host, pi_dev, pi_size, cudaMemcpyDeviceToHost);
  int j;
  for (j=0; j < (THREADS_PER_BLOCK*NUM_BLOCKS); j++) 
	  pi += pi_host[j];

  runtime = stopTimer();
  printf("pi = %.12f\n", pi);
  printf("Total wall time: %f ms\n", runtime/1000.0);
  free(pi_host);
  cudaFree(pi_dev);

  return 0;
}

