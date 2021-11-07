#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else

#include <CL/cl.h>

#endif

int main() {
    cl_int ret;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id *device_id = NULL;
    cl_uint ret_num_devices;

    /* Get Platform and Device Info */
    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    if (ret_num_platforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        return EXIT_FAILURE;
    }
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &ret_num_devices);
    printf("Found %d platforms, %d devices.\n", ret_num_platforms, ret_num_devices);
    if (ret_num_platforms == 0 || ret_num_devices == 0) {
        return EXIT_FAILURE;
    }
    char cBuffer[1024];
    clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    printf("Name: %s\n", cBuffer);

    device_id = malloc( sizeof(cl_device_id)*ret_num_devices);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, ret_num_devices, device_id, &ret_num_devices);
    for (int i=0; i<ret_num_devices; i++) {
        clGetDeviceInfo(device_id[0], CL_DEVICE_NAME, sizeof(cBuffer), cBuffer, NULL);
        printf("Device %d: %s, ", i, cBuffer);
        clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, sizeof(cBuffer), cBuffer, NULL);
        printf("%s\n", cBuffer);
        clGetDeviceInfo(device_id[i], CL_DRIVER_VERSION, sizeof(cBuffer), cBuffer, NULL);
        printf("   Driver version: %s\n", cBuffer);
        clGetDeviceInfo(device_id[i], CL_DEVICE_VERSION, sizeof(cBuffer), cBuffer, NULL);
        printf("   Device version: %s\n", cBuffer);
        clGetDeviceInfo(device_id[i], CL_DEVICE_OPENCL_C_VERSION, sizeof(cBuffer), cBuffer, NULL);
        printf("   OpenCL C version: %s\n", cBuffer);
        cl_uint ncpu;
        clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &ncpu, NULL);
        printf("   Compute units: %u\n", (unsigned)ncpu);
	unsigned mgs=0;
        clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &mgs, NULL);
        printf("   Compute workgroup: %u\n", (unsigned)mgs);
    }


    return EXIT_SUCCESS;
}

