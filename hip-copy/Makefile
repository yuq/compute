#!makefile

all: test.out

test.out: main.c
	gcc -g -I/opt/rocm/hip/include -D__HIP_PLATFORM_HCC__ -o $@ $< -L/opt/rocm/hip/lib -lamdhip64

clean:
	rm *.o *.out
