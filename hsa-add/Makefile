#!makefile

CLANG := clang-7

all: test.out hello.co

test.out: main.c
	gcc -I/opt/rocm/hsa/include/hsa -o $@ $< -L/opt/rocm/lib -lhsa-runtime64

hello.co: hello.o
	$(CLANG) -target amdgcn--amdhsa -o $@ $<

hello.o: hello.s
	$(CLANG) -x assembler -target amdgcn--amdhsa -mcpu=gfx906 -c -o $@ $<

clean:
	rm -rf *.o *.out *.co
