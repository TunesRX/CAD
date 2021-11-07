#
# Created by vad on 16/10/21.
# DI FCTUNL
#

CFLAGS	:= 

PROGS = add add-cuda add-cl

.PHONY:	all

all:	$(PROGS)

add:    add.c vsize.h
	cc $(CFLAGS) -o $@ $<

add-cuda:	add-cuda.cu vsize.h
	nvcc $(CFLAGS) -o $@ $<

add-cl:	add-cl.c vsize.h
	cc $(CFLAGS) -DCL_TARGET_OPENCL_VERSION=200 -o $@ $< -lOpenCL

clean:
	rm -f $(PROGS)
