#!makefile

MPI_PATH ?= /home/yuq/program/ompi-dev

test.out: main.cpp
	$(MPI_PATH)/bin/mpic++ -I/opt/rocm/hip/include -I$(MPI_PATH)/include -D__HIP_PLATFORM_HCC__ -std=c++11 -g -o $@ $< -L$(MPI_PATH)/lib -lmpi -L/opt/rocm/lib/ -lhip_hcc

run: test.out
	PATH=$(MPI_PATH)/bin:$(PATH) LD_LIBRARY_PATH=$(MPI_PATH)/lib:$(LD_LIBRARY_PATH) \
	mpirun -np 2 --allow-run-as-root -mca pml ucx -x UCX_TLS=sm,rocm_cpy,rocm_gdr,rocm_ipc test.out

clean:
	rm -rf *.o *.out
