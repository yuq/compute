#!/bin/bash

test.out: main.c
	gcc -I/opt/rocm/include -o $@ $< -L/opt/rocm/lib -lhsa-runtime64
