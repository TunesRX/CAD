#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define NUM_STEPS1  50000000
#define NUM_STEPS2  30000000

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

void eCalc( double *e, long niter){
    double factorial;
    *e = 1.0; factorial = 1.0;
     for(long i = 1; i < niter; i++){
          factorial  *= i;
          *e += 1.0/factorial;
    }
}


double f(double x) {
    return 1.0/(x*x + 1.0);
}

void piCalc( double *pi, long niter){
     double step; double x;
     step= 1.0/(double)niter;
     *pi = 0.0;
     for(long i = 0; i < niter; i++){
		 x = (double)i*step;

        *pi = *pi + f(x);
     }
     *pi = 4.0* (*pi)*step;
}

int main(int argc, char *argv[])
{
  double e, pi;
  float runtime;
  long num_steps1, num_steps2;

  if (argc==1) {
    num_steps1=NUM_STEPS1;
    num_steps2=NUM_STEPS2;
  } else if (argc==2) {
    num_steps1 = num_steps2 = atoi(argv[1]);
  } else {
    fprintf(stderr, "usage: e_and_pi [num steps]\n");
    return 1;
  }
 
  printf("Started...\n");
  startTimer();

  /* First we calculate e from its taylor expansion */
  eCalc( &e, num_steps1); 
  /* Then we calculate pi from integral 4/(1+x*x) */
  piCalc( &pi, num_steps2);

  runtime = stopTimer();
  printf("e = %.12f\n", e);
  printf("pi = %.12f\n", pi);
  printf("Total wall time: %f ms\n", runtime/1000.0);

  return 0;
}
