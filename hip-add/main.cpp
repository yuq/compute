#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>

#include <hip/hip_runtime.h>

#define VECTOR_SIZE 0x10000
#define BLOCK_SIZE 512

__global__ void vec_add(float *a, float *b, float *c, int n)
{
	//int id = hipThreadIdx_x;
	int id = blockIdx.x*blockDim.x+threadIdx.x;
	if (id < n)
		c[id] = a[id] + b[id];
}

int main(void)
{
	float *A_h, *B_h, *C_h;
	float *A_d, *B_d, *C_d;
	int size = VECTOR_SIZE;
	int nb = size * sizeof(double);

	A_h = (float *)malloc(nb);
	B_h = (float *)malloc(nb);
	C_h = (float *)malloc(nb);
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

	//hipDeviceSynchronize();

	assert(hipMemcpy(C_h, C_d, nb, hipMemcpyDeviceToHost) == hipSuccess);

	bool success = true;
	for (int i = 0; i < size; i++) {
		if (C_h[i] != 2.0) {
			fprintf(stderr, "check result fail %d %f\n", i, C_h[i]);
			success = false;
			break;
		}
	}
	if (success)
		printf("success\n");

	free(A_h);
	free(B_h);
	free(C_h);
	hipFree(A_d);
	hipFree(B_d);
	hipFree(C_d);
	return 0;
}
