
all:	devq cuda-devq

devq:	devq.c
	cc -g -DCL_TARGET_OPENCL_VERSION=220 -o devq devq.c -lOpenCL
#	cc devq.c /System/Library/Frameworks/OpenCL.framework/OpenCL

cuda-devq:	cuda-devq.cu
	nvcc -o $@ $<

clean:
	rm -f *~ a.out  devq cuda-devq
