#!makefile

all: test.out

test.out: main.c
	gcc -I/opt/rocm/hip/include -D__HIP_PLATFORM_HCC__ -o $@ $< -L/opt/rocm/hip/lib -lhip_hcc

clean:
	rm *.o *.out
