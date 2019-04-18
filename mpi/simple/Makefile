#!makefile

#MPI_INC=$(shell pkg-config --cflags mpi)
#MPI_LIB=$(shell pkg-config --libs mpi) -lmpi_cxx
MPI_INC=-I$(MPI_PATH)/include
MPI_LIB=-L$(MPI_PATH)/lib -lmpi

test.out: main.cpp
	hipcc $(MPI_INC) -g --amdgpu-target=gfx900 --amdgpu-target=gfx906 -o $@ $< $(MPI_LIB)

clean:
	rm -rf *.o *.out
