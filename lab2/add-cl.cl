/*
*  Created by vad on 16/10/21.
*  DI FCTUNL
*/

#include "vsize.h"

/* SAXPY BLAS like function
 * single precision X=a*X+Y  (X and Y vectors)
 * out: result in X
 */
__kernel void saxpy( float a, __global float *X, __global float *Y) {
    int i = get_global_id(0);
	/* TODO: */
}
