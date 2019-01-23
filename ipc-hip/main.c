#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <hip/hip_runtime.h>

#define BUFF_SIZE (1 << 30)

static void parent(int fd)
{
    assert(hipSetDevice(0) == hipSuccess);

    void *src = NULL;
    assert(hipMalloc(&src, BUFF_SIZE) == hipSuccess);
    //memset(src, 0x23, BUFF_SIZE);

    hipIpcMemHandle_t ipc;
    assert(read(fd, &ipc, sizeof(ipc)) == sizeof(ipc));

    void *dst = NULL;
    assert(hipIpcOpenMemHandle(&dst, ipc, hipIpcMemLazyEnablePeerAccess) == hipSuccess);

    struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    //assert(hipMemcpyDtoD(dst, src, BUFF_SIZE) == hipSuccess);
    hipStream_t stream;
    assert(hipStreamCreate(&stream) == hipSuccess);
    assert(hipMemcpyDtoDAsync(dst, src, BUFF_SIZE, stream) == hipSuccess);
    assert(hipDeviceSynchronize() == hipSuccess);

    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE / (end - start);
	printf("copy rate %f GB/s\n", rate);

    assert(hipIpcCloseMemHandle(dst) == hipSuccess);
}

static void child(int fd)
{
    assert(hipSetDevice(1) == hipSuccess);

    void *ptr = NULL;
    assert(hipMalloc(&ptr, BUFF_SIZE) == hipSuccess);

    hipIpcMemHandle_t ipc;
    assert(hipIpcGetMemHandle(&ipc, ptr) == hipSuccess);

    assert(write(fd, &ipc, sizeof(ipc)) == sizeof(ipc));

	sleep(5);
}

int main(int argc, char **argv)
{
	int pipefd[2];
	assert(!pipe(pipefd));

	pid_t pid = fork();
	if (pid)
		parent(pipefd[0]);
	else
		child(pipefd[1]);

	return 0;
}
