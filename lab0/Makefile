
CC=cc

#CFLAGS=-g # debug
# CFLAGS=-pg    # profile gprof
CFLAGS=-O3 -fopt-info  #optimize and get info about it


e_and_pi:   e_and_pi.c
	$(CC) $(CFLAGS)  -o $@  $<

e_and_pi_par:   e_and_pi_par.c
	$(CC) $(CFLAGS) -fopenmp -o $@  $<

all:    e_and_pi e_and_pi_par


.PHONY: clean
clean:  
	rm -f *.o e_and_pi e_and_pi_par
