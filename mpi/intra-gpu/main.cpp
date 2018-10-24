#include <stdio.h>
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

#define BUFFER_SIZE 0x100000
char h_buffer[BUFFER_SIZE] = "hello";

void sender(void)
{
	char *d_buffer;
	CHECK(hipMalloc(&d_buffer, sizeof(h_buffer)));
	CHECK(hipMemcpy(d_buffer, h_buffer, sizeof(h_buffer), hipMemcpyHostToDevice));

	MPI_Send(d_buffer, sizeof(h_buffer), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

void receiver(void)
{
	char *d_buffer;
	CHECK(hipMalloc(&d_buffer, sizeof(h_buffer)));
	
	MPI_Status status;
	MPI_Recv(d_buffer, sizeof(h_buffer), MPI_CHAR, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	char result[BUFFER_SIZE];
	CHECK(hipMemcpy(result, d_buffer, sizeof(h_buffer), hipMemcpyDeviceToHost));
	printf("result: %s\n", result);
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
