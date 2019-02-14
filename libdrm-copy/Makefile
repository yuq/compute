#!makefile

LIBDRM_CFLAGS=$(shell pkg-config --cflags libdrm_amdgpu)
LIBDRM_LDFLAGS=$(shell pkg-config --libs libdrm_amdgpu)

all: test.out

test.out: main.c
	gcc -g $(LIBDRM_CFLAGS) -o $@ $< $(LIBDRM_LDFLAGS)

clean:
	rm *.o *.out
