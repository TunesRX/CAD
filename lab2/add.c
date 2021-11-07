//
// Created by vad on 16/10/21.
// DI FCTUNL
//

#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "vsize.h"

// intVec - initialize all v with val
void initVec(float *v, float val) {
    for (int i=0; i<VSIZE; i++)
        v[i]=val;
}

// checkVec - check that all v has val
void checkVec(float *v, float val) {
    for (int i=0; i<VSIZE; i+=1000) {
        //printf("[%d] -> %f\n", i, v[i]);
        assert(v[i]==val);
    }
}

/* SAXPY BLAS like function
 * single precision X=a*X+Y  (X and Y vectors)
 * out: result in X
 */
void saxpy(float a, float *X, float *Y) {
    for (int i=0; i<VSIZE; i++)
        X[i]=a*X[i]+Y[i];
}


/******* MAIN ******/

int main() {
    printf("starting...\n");
    float *a = malloc(VSIZE*sizeof(float));
    float *b = malloc(VSIZE*sizeof(float));
    if ( a==NULL || b==NULL ) {
        fprintf(stderr,"No mem!\n");
        return EXIT_FAILURE;
    }
    initVec(a, 1.0);
    initVec(b, 2.0);

    clock_t t=clock();
    saxpy(2.0, a, b);
    t=clock()-t;
    printf("time: %f ms\n", ((double)t/(double)CLOCKS_PER_SEC)*1000.0);

    checkVec(a, 4.0); // 2*1+2 == 4
    printf("OK\n");

    return EXIT_SUCCESS;
}
