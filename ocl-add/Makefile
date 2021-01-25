#!makefile

INC_PATH ?= /opt/rocm/opencl/include
LIB_PATH ?= /opt/rocm/lib

all: test.out

test.out: main.c
	gcc -g -I$(INC_PATH) -o $@ $< -L$(LIB_PATH) -lOpenCL

clean:
	rm *.o *.out
