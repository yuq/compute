#!makefile

LIBS:=-L/opt/rocm/hip/lib -lhip_hcc -lstdc++

test.out: main.o test-c.o test-hip.o
	gfortran -o $@ $^ $(LIBS)

main.o: main.F
	gfortran -c -o $@ $<

test-c.o: test-c.c
	gcc -c -o $@ $<

test-hip.o: test-hip.cpp
	hipcc -c -fno-gpu-rdc -o $@ $<

clean:
	rm -rf *.o *.out
