#!makefile

test.out: main.cpp
	hipcc -g -I/opt/rocm/hipblas/include -o $@ $< -L/opt/rocm/hipblas -lhipblas
