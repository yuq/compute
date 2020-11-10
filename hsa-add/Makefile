#!makefile

CLANG := clang-7

INC_PATH ?= /opt/rocm/hsa/include/hsa
LIB_PATH ?= /opt/rocm/lib

all: test.out hello.co

test.out: main.c
	gcc -g -I$(INC_PATH) -o $@ $< -L$(LIB_PATH) -lhsa-runtime64

hello.co: hello.o
	$(CLANG) -target amdgcn--amdhsa -o $@ $<

hello.o: hello.s
	$(CLANG) -x assembler -target amdgcn--amdhsa -mcpu=gfx900 -c -o $@ $<

clean:
	rm -rf *.o *.out *.co
