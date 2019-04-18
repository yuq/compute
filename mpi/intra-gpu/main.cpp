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

#define BUFFER_SIZE 0x100000

void sender(void)
{
    void *buffer;
    assert(!posix_memalign(&buffer, 0x1000, BUFFER_SIZE));
    memset(buffer, 0x23, BUFFER_SIZE);

    assert(hipHostRegister(buffer, BUFFER_SIZE, hipHostRegisterDefault) == hipSuccess);

    memset(buffer, 0x23, BUFFER_SIZE);

	MPI_Send(buffer, BUFFER_SIZE, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
}

void receiver(void)
{
    unsigned char *buffer;
    assert(!posix_memalign((void **)&buffer, 0x1000, BUFFER_SIZE));

    assert(hipHostRegister((void *)buffer, BUFFER_SIZE, hipHostRegisterDefault) == hipSuccess);

    memset(buffer, 0x23, BUFFER_SIZE);

	MPI_Status status;
	MPI_Recv(buffer, BUFFER_SIZE, MPI_CHAR, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    for (int i = 0; i < BUFFER_SIZE; i++) {
        if (buffer[i] != 0x23) {
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
