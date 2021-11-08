

.PHONY:	all

all:	cuda

main:	main.c 
	cc -g -o $@ $< 

# TODO:
cuda:	cuda.cu
	nvcc -g -o $@ $<


.PHONY:	clean
clean:
	rm -f cuda out.ppm
