#!makefile

test.out: main.o test-c.o test-hip.o
	hipcc -o $@ $^ -lhipblas -lgfortran

main.o: main.F
	gfortran -c -o $@ $<

test-c.o: test-c.c
	gcc -c -o $@ $<

test-hip.o: test-hip.cpp
	hipcc -c -o $@ $<

clean:
	rm -rf *.o *.out
