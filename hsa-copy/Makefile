#!makefile

INC_PATH ?= /opt/rocm/hsa/include/hsa
LIB_PATH ?= /opt/rocm/lib

all: test.out

test.out: main.c
	gcc -g -I$(INC_PATH) -o $@ $< -L$(LIB_PATH) -lhsa-runtime64

clean:
	rm *.o *.out
