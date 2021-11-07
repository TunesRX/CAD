//
// Created by vad on 16/10/21.
// DI FCTUNL
//

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "vsize.h"

/*------------Open CL auxiliary functions--------------*/
char *cl_errstr(cl_int err) {
#define CaseReturnString(x) case x: return #x;
    switch (err) {
        CaseReturnString(CL_SUCCESS                        )
        CaseReturnString(CL_DEVICE_NOT_FOUND               )
        CaseReturnString(CL_DEVICE_NOT_AVAILABLE           )
        CaseReturnString(CL_COMPILER_NOT_AVAILABLE         )
        CaseReturnString(CL_MEM_OBJECT_ALLOCATION_FAILURE  )
        CaseReturnString(CL_OUT_OF_RESOURCES               )
        CaseReturnString(CL_OUT_OF_HOST_MEMORY             )
        CaseReturnString(CL_PROFILING_INFO_NOT_AVAILABLE   )
        CaseReturnString(CL_MEM_COPY_OVERLAP               )
        CaseReturnString(CL_IMAGE_FORMAT_MISMATCH          )
        CaseReturnString(CL_IMAGE_FORMAT_NOT_SUPPORTED     )
        CaseReturnString(CL_BUILD_PROGRAM_FAILURE          )
        CaseReturnString(CL_MAP_FAILURE                    )
        CaseReturnString(CL_MISALIGNED_SUB_BUFFER_OFFSET   )
        CaseReturnString(CL_COMPILE_PROGRAM_FAILURE        )
        CaseReturnString(CL_LINKER_NOT_AVAILABLE           )
        CaseReturnString(CL_LINK_PROGRAM_FAILURE           )
        CaseReturnString(CL_DEVICE_PARTITION_FAILED        )
        CaseReturnString(CL_KERNEL_ARG_INFO_NOT_AVAILABLE  )
        CaseReturnString(CL_INVALID_VALUE                  )
        CaseReturnString(CL_INVALID_DEVICE_TYPE            )
        CaseReturnString(CL_INVALID_PLATFORM               )
        CaseReturnString(CL_INVALID_DEVICE                 )
        CaseReturnString(CL_INVALID_CONTEXT                )
        CaseReturnString(CL_INVALID_QUEUE_PROPERTIES       )
        CaseReturnString(CL_INVALID_COMMAND_QUEUE          )
        CaseReturnString(CL_INVALID_HOST_PTR               )
        CaseReturnString(CL_INVALID_MEM_OBJECT             )
        CaseReturnString(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
        CaseReturnString(CL_INVALID_IMAGE_SIZE             )
        CaseReturnString(CL_INVALID_SAMPLER                )
        CaseReturnString(CL_INVALID_BINARY                 )
        CaseReturnString(CL_INVALID_BUILD_OPTIONS          )
        CaseReturnString(CL_INVALID_PROGRAM                )
        CaseReturnString(CL_INVALID_PROGRAM_EXECUTABLE     )
        CaseReturnString(CL_INVALID_KERNEL_NAME            )
        CaseReturnString(CL_INVALID_KERNEL_DEFINITION      )
        CaseReturnString(CL_INVALID_KERNEL                 )
        CaseReturnString(CL_INVALID_ARG_INDEX              )
        CaseReturnString(CL_INVALID_ARG_VALUE              )
        CaseReturnString(CL_INVALID_ARG_SIZE               )
        CaseReturnString(CL_INVALID_KERNEL_ARGS            )
        CaseReturnString(CL_INVALID_WORK_DIMENSION         )
        CaseReturnString(CL_INVALID_WORK_GROUP_SIZE        )
        CaseReturnString(CL_INVALID_WORK_ITEM_SIZE         )
        CaseReturnString(CL_INVALID_GLOBAL_OFFSET          )
        CaseReturnString(CL_INVALID_EVENT_WAIT_LIST        )
        CaseReturnString(CL_INVALID_EVENT                  )
        CaseReturnString(CL_INVALID_OPERATION              )
        CaseReturnString(CL_INVALID_GL_OBJECT              )
        CaseReturnString(CL_INVALID_BUFFER_SIZE            )
        CaseReturnString(CL_INVALID_MIP_LEVEL              )
        CaseReturnString(CL_INVALID_GLOBAL_WORK_SIZE       )
        CaseReturnString(CL_INVALID_PROPERTY               )
        CaseReturnString(CL_INVALID_IMAGE_DESCRIPTOR       )
        CaseReturnString(CL_INVALID_COMPILER_OPTIONS       )
        CaseReturnString(CL_INVALID_LINKER_OPTIONS         )
        CaseReturnString(CL_INVALID_DEVICE_PARTITION_COUNT )
        default: return "Unknown OpenCL error code";
    }
}

// checkError - check Open CL error and print message if needed
void checkError(cl_int err, const char *msg) {
    if (err!=CL_SUCCESS) {
        fprintf(stderr,"(%d %s) %s\n", err, cl_errstr(err), msg);
        exit(1);
    }
}

// checkBuild - check kernel build and print error if needed
void checkBuild(cl_int err, cl_program program, cl_device_id device, const char *msg) {
    if (err != CL_SUCCESS) {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        printf("Error in kernel: %s", buildLog);;
        clReleaseProgram(program);
        exit(1);
    }
}

// myCLInit - initialize OpenCL framework and get GPU device id
cl_device_id myCLInit() {
    cl_int ret;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id device_id;
    cl_uint ret_num_devices;
    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret_num_platforms == 0 || ret_num_devices == 0) {
        fprintf(stderr, "No device found\n");
        exit(EXIT_FAILURE);
    }
    return device_id;
}

// myBuildKernel - build a kernel from am external file
cl_kernel myBuildKernel(cl_context context, cl_device_id device, char *filename, char *funcname) {
    FILE *f=fopen(filename,"r");
    if (f==NULL) { fprintf(stderr,"can't read file %s\n", filename); exit(1); }
    fseek(f, 0L, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0L, SEEK_SET);

    char *src = malloc(fsize+1);
    fread(src, 1, fsize, f);
    src[fsize]='\0';
    cl_int ret;

    cl_program program = clCreateProgramWithSource(context, 1,(const char **)&src, NULL, &ret);
    /* Build Kernel Program */
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    checkBuild(ret, program, device, "Can't build kernel");
    /* Create OpenCL Kernel */
    cl_kernel kernel = clCreateKernel(program, funcname, &ret);
    checkError(ret, "Can't create kernel.\n");
    return kernel;
}
/*--------------------------*/

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

/******* MAIN ******/

int main(int argc, char*argv[]) {
    size_t localWS=GROUPSIZE;
	if (argc==2) localWS=atoi(argv[1]);
	else if (argc>2) {
        fprintf(stderr,"usage: %s [workgroup_size]\n", argv[0]);
	    return EXIT_FAILURE;
    }
    printf("starting...\n");
    size_t globalWS=localWS*((VSIZE+localWS-1)/localWS);  //total num threads multiple of localWS
    printf("Using %zu threads, %u groups of %u threads\n", globalWS, globalWS/localWS, localWS);

    float *a = malloc(VSIZE*sizeof(float));
    float *b = malloc(VSIZE*sizeof(float));
    if ( a==NULL || b==NULL ) {
        fprintf(stderr,"No mem!\n");
        return EXIT_FAILURE;
    }
    initVec(a, 1.0);
    initVec(b, 2.0);

    cl_int ret;
    cl_device_id device = myCLInit();
    cl_context context = clCreateContext(0, 1, &device, NULL, NULL, &ret);
	cl_command_queue comQueue = clCreateCommandQueueWithProperties(context, device, NULL, &ret);
    cl_kernel kernel = myBuildKernel(context, device, "add-cl.cl", "saxpy");

    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  VSIZE*sizeof(float), NULL, &ret);
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  VSIZE*sizeof(float), NULL, &ret);
    checkError(ret,"No GPU mem!\n");

    ret = clEnqueueWriteBuffer(comQueue, a_mem, CL_TRUE, 0,
                               VSIZE*sizeof(float), a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(comQueue, b_mem, CL_TRUE, 0,
                               VSIZE*sizeof(float), b, 0, NULL, NULL);

    // Set the arguments of the kernel and run kernel
    float c=2.0;

    clock_t t=clock();
    // call saxpy(c, a, b)
    ret = clSetKernelArg(kernel, 0, sizeof(float), (void*)&c);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&a_mem);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_mem);
    ret = clEnqueueNDRangeKernel(comQueue, kernel, 1, NULL,
                                 &globalWS, &localWS, 0, NULL, NULL);
    t = clock()-t;
    printf("time: %f ms\n", (t/(double)CLOCKS_PER_SEC)*1000.0);
    checkError(ret,"Problems executing kernel\n");

    ret = clEnqueueReadBuffer(comQueue, a_mem, CL_TRUE, 0,
                              VSIZE*sizeof(float), a, 0, NULL, NULL);

    checkVec(a, 4.0); // 2*1+2 == 4
    printf("OK\n");

    return EXIT_SUCCESS;
}
