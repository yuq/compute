#!makefile

test.out: main.cpp
	hipcc -g -o $@ $< -lhipblas

clean:
	rm -rf *.o *.out
