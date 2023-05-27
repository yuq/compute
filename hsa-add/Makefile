#!makefile

CLANG ?= clang-15

INC_PATH ?= /opt/rocm/hsa/include/hsa
LIB_PATH ?= /opt/rocm/lib

all: test.out hello.gfx900.co hello.gfx1030.co

test.out: main.c
	gcc -g -I$(INC_PATH) -o $@ $< -L$(LIB_PATH) -lhsa-runtime64

hello.gfx900.co: hello.gfx900.o
	$(CLANG) -target amdgcn--amdhsa -o $@ $<

hello.gfx900.o: hello.gfx900.s
	$(CLANG) -x assembler -target amdgcn--amdhsa -mcpu=gfx900 -mcode-object-version=2 -c -o $@ $<

hello.gfx1030.co: hello.gfx1030.o
	$(CLANG) -target amdgcn--amdhsa -o $@ $<

hello.gfx1030.o: hello.gfx1030.s
	$(CLANG) -x assembler -target amdgcn--amdhsa -mcpu=gfx1030 -mcode-object-version=2 -c -o $@ $<

clean:
	rm -rf *.o *.out *.co
