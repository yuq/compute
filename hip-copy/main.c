#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <hip/hip_runtime.h>

#define BUFF_SIZE (1 << 25)

void test(int p2d, int use_hip)
{
    void *src, *dst;
    int kind;
    if (p2d) {
        kind = hipMemcpyHostToDevice;
        assert(hipMalloc(&dst, BUFF_SIZE) == hipSuccess);
        assert(hipMallocHost(&src, BUFF_SIZE) == hipSuccess);
    }
    else {
        kind = hipMemcpyDeviceToHost;
        assert(hipMalloc(&src, BUFF_SIZE) == hipSuccess);
        assert(hipMallocHost(&dst, BUFF_SIZE) == hipSuccess);
    }

    struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    if (use_hip)
        hipMemcpy(dst, src, BUFF_SIZE, kind);
    else
        memcpy(dst, src, BUFF_SIZE);

    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE / (end - start);
	printf("%s %s copy rate %f GB/s\n", p2d ? "P2D" : "D2P",
           use_hip ? "hip" : "mem", rate);

    if (p2d) {
        hipFree(dst);
        hipFreeHost(src);
    }
    else {
        hipFree(src);
        hipFreeHost(dst);
    }
}

int main(void)
{
    test(0, 0);
    test(1, 0);
    test(0, 1);
    test(1, 1);
    return 0;
}
