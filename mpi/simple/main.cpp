#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <mpi.h>

#include <hip/hip_runtime.h>

#define VECTOR_SIZE 0x10000
#define BLOCK_SIZE 512

__global__ void vec_add(double *a, double *b, double *c, int n)
{
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < n)
		c[id] = a[id] + b[id];
}

void do_compute(void)
{
	double *A_h, *B_h, *C_h;
	double *A_d, *B_d, *C_d;
	int size = VECTOR_SIZE;
	int nb = size * sizeof(double);

	A_h = (double *)malloc(nb);
	B_h = (double *)malloc(nb);
	C_h = (double *)malloc(nb);
	assert(A_h && B_h && C_h);
  
	for (int i = 0; i < size; i++) {
		A_h[i] = B_h[i] = 1;
		C_h[i] = 0;
	}

	assert(hipMalloc(&A_d, nb) == hipSuccess);
	assert(hipMalloc(&B_d, nb) == hipSuccess);
	assert(hipMalloc(&C_d, nb) == hipSuccess);

	assert(hipMemcpy(A_d, A_h, nb, hipMemcpyHostToDevice) == hipSuccess);
	assert(hipMemcpy(B_d, B_h, nb, hipMemcpyHostToDevice) == hipSuccess);

	hipLaunchKernelGGL(vec_add, dim3(VECTOR_SIZE/BLOCK_SIZE), dim3(BLOCK_SIZE),
			   0, 0, A_d, B_d, C_d, size);

	assert(hipMemcpy(C_h, C_d, nb, hipMemcpyDeviceToHost) == hipSuccess);
	for (int i = 0; i < size; i++)
		assert(C_h[i] == 2);

	free(A_h);
	free(B_h);
	free(C_h);
	hipFree(A_d);
	hipFree(B_d);
	hipFree(C_d);
}

void gpu_main(void)
{
	for (int i = 0; i < 1000; i++) {
		do_compute();

		if (!(i % 100))
			printf("i=%d\n", i);
	}
	printf("done\n");
}

int main(int argc, char **argv)
{
	int pid=-1, np=-1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	MPI_Barrier(MPI_COMM_WORLD);

	if (!pid)
		gpu_main();

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}
