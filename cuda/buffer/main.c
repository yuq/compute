#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>

#define BUFF_SIZE 0x100000

static int hello(void)
{
  printf("hello\n");
}

int main(void)
{
  void *d = NULL;
  assert(cudaMalloc(&d, BUFF_SIZE) == cudaSuccess);

  void *h = NULL;
  assert(cudaMallocHost(&h, BUFF_SIZE) == cudaSuccess);

  void *m = NULL;
  assert(cudaMallocManaged(&m, BUFF_SIZE, cudaMemAttachGlobal) == cudaSuccess);

  printf("d=%p h=%p m=%p\n", d, h, m);

  hello();
  return 0;
}
