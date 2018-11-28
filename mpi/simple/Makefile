#!makefile

MPI_INC=$(shell pkg-config --cflags mpi)
MPI_LIB=$(shell pkg-config --libs mpi) -lmpi_cxx

test.out: main.cpp
	hipcc $(MPI_INC) -g -o $@ $< $(MPI_LIB)

clean:
	rm -rf *.o *.out
