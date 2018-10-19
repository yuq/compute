#!makefile

MPI_PATH=/home/yuq/program/ompi

test.out: main.cpp
	hipcc -I$(MPI_PATH)/include -o $@ $< -L$(MPI_PATH)/lib -lmpi

run: test.out
	PATH=$(MPI_PATH)/bin:$(PATH) LD_LIBRARY_PATH=$(MPI_PATH)/lib:$(LD_LIBRARY_PATH) \
	mpirun -np 2 -mca pml ucx test.out

clean:
	rm -rf *.o *.out
