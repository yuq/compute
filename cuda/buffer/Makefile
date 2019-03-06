#!makefile

all: test.out

test.out: main.c
	gcc -g -I/usr/local/cuda/include -o $@ $< -L/usr/local/cuda/lib64 -lcudart

clean:
	rm *.o *.out
