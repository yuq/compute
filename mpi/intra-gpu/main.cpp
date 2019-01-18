#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include <mpi.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd) \
{ \
    hipError_t error  = cmd; \
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", \
		hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define BUFFER_SIZE 0x400000

void sender(void)
{
	char *h_buffer = (char *)malloc(BUFFER_SIZE);
	assert(h_buffer);
	memset(h_buffer, 0x23, BUFFER_SIZE);

	char *d_buffer;
	CHECK(hipMalloc(&d_buffer, BUFFER_SIZE));
	CHECK(hipMemcpy(d_buffer, h_buffer, BUFFER_SIZE, hipMemcpyHostToDevice));

	MPI_Send(d_buffer, BUFFER_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

void receiver(void)
{
	char *d_buffer;
	CHECK(hipMalloc(&d_buffer, BUFFER_SIZE));
	
	MPI_Status status;
	MPI_Recv(d_buffer, BUFFER_SIZE, MPI_CHAR, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	char *result = (char *)malloc(BUFFER_SIZE);
	assert(result);

	CHECK(hipMemcpy(result, d_buffer, BUFFER_SIZE, hipMemcpyDeviceToHost));
    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (result[i] != 0x23) {
            printf("fail\n");
            return;
        }
    }
    printf("pass\n");
}

int main(int argc, char **argv)
{
	hipDeviceProp_t props;
	CHECK(hipGetDeviceProperties(&props, 0));
	printf("info: running on device %s\n", props.name);

	int pid=-1, np=-1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &pid);
	MPI_Comm_size(MPI_COMM_WORLD, &np);

	MPI_Barrier(MPI_COMM_WORLD);

	if (pid)
		sender();
	else
		receiver();

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}
