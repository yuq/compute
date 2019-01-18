#!makefile

MPI_PATH ?= /home/yuq/program/ompi-dev

test.out: main.cpp
	hipcc -I$(MPI_PATH)/include -g -o $@ $< -L$(MPI_PATH)/lib -lmpi

run: test.out
	PATH=$(MPI_PATH)/bin:$(PATH) LD_LIBRARY_PATH=$(MPI_PATH)/lib:$(LD_LIBRARY_PATH) \
	mpirun -np 2 --allow-run-as-root -mca pml ucx -x UCX_TLS=mm,rocm_ipc test.out

clean:
	rm -rf *.o *.out
