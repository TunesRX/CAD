all:	pi pi_cuda pi_calc


pi:	pi.c
	cc -o pi pi.c

pi_cuda:	pi_cuda.cu
	nvcc -o $@ $<

pi_calc:	pi_calc.c
	c++ -DCL_TARGET_OPENCL_VERSION=220 -g -o $@ $< -lOpenCL


clean:
	rm -f *~ pi pi_cuda pi_calc *.o
