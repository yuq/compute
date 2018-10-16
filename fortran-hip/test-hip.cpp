#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hipblas.h>

#define CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
	  }\
}

extern "C" {
void test_hip_(void);
}

void test_hip_(void)
{
  hipDeviceProp_t props;
  CHECK(hipGetDeviceProperties(&props, 0));
  printf ("info: running on device %s\n", props.name);

  double *A_h, *B_h, *C_h;
  double *A_d, *B_d, *C_d;
  int size = 16;
  int nb = size * size * sizeof(double);

  A_h = (double *)malloc(nb);
  B_h = (double *)malloc(nb);
  C_h = (double *)malloc(nb);
  
  for (int i = 0; i < size * size; i++)
    A_h[i] = B_h[i] = C_h[i] = 1;

  CHECK(hipMalloc(&A_d, nb));
  CHECK(hipMalloc(&B_d, nb));
  CHECK(hipMalloc(&C_d, nb));

  CHECK(hipMemcpy(A_d, A_h, nb, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(B_d, B_h, nb, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(C_d, C_h, nb, hipMemcpyHostToDevice));

  hipblasHandle_t handle;
  hipblasCreate(&handle);

  hipblasOperation_t transa = HIPBLAS_OP_N;
  hipblasOperation_t transb = HIPBLAS_OP_N;
  double alpha = 1, beta = 1;
  
  hipblasDgemm(handle, transa, transb, size, size, size,
	       &alpha, A_d, size, B_d, size, &beta, C_d, size);

  CHECK(hipMemcpy(C_h, C_d, nb, hipMemcpyDeviceToHost));

  for (int i = 0; i < size * size; i++)
    printf("%f ", C_h[i]);
  printf("\n");
}
