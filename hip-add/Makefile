#!makefile

all: test.out

test.out: main.cpp
	hipcc -g --amdgpu-target=gfx900 --amdgpu-target=gfx1030 -o $@ $< 

clean:
	rm *.o *.out
