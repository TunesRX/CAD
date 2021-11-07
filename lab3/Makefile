

.PHONY:	all

all:	main

main:	main.c 
	cc -g -o $@ $< 

# TODO:
cuda:	cuda.cu
	nvcc -g -o $@ $<


.PHONY:	clean
clean:
	rm -f main out.ppm
