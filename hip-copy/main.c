#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <hip/hip_runtime.h>

#define BUFF_SIZE (1 << 25)

void test_dp(int p2d, int use_hip)
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
    assert(hipMemset(src, 0, BUFF_SIZE) == hipSuccess);
    assert(hipMemset(dst, 0, BUFF_SIZE) == hipSuccess);

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

void test_dd(int cross_gpu)
{
    void *src, *dst;

    hipSetDevice(cross_gpu);
    assert(hipMalloc(&dst, BUFF_SIZE) == hipSuccess);
    hipSetDevice(0);
    assert(hipMalloc(&src, BUFF_SIZE) == hipSuccess);
    assert(hipMemset(src, 0, BUFF_SIZE) == hipSuccess);
    assert(hipMemset(dst, 0, BUFF_SIZE) == hipSuccess);

    hipStream_t stream;
    assert(hipStreamCreate(&stream) == hipSuccess);

    struct timespec tv1, tv2;
    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    assert(hipMemcpyDtoDAsync(dst, src, BUFF_SIZE, stream) == hipSuccess);
    assert(hipDeviceSynchronize() == hipSuccess);

    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    double start = tv1.tv_sec;
    start = start * 1e9 + tv1.tv_nsec;

    double end = tv2.tv_sec;
    end = end * 1e9 + tv2.tv_nsec;

    double rate = BUFF_SIZE / (end - start);
    printf("%s D2D copy rate %f GB/s\n", cross_gpu ? "inter" : "intra", rate);

    hipFree(dst);
    hipFree(src);
}

void test_copy_one(const char *name, void *dst, void *src)
{
    struct timespec tv1, tv2;
    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    hipMemcpy(dst, src, BUFF_SIZE, hipMemcpyHostToHost);

    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    double start = tv1.tv_sec;
    start = start * 1e9 + tv1.tv_nsec;

    double end = tv2.tv_sec;
    end = end * 1e9 + tv2.tv_nsec;

    double rate = BUFF_SIZE / (end - start);
    printf("%s copy rate %f GB/s\n", name, rate);
}

void test_pp(void)
{
    void *src, *dst;

    assert(hipMallocHost(&dst, BUFF_SIZE) == hipSuccess);
    assert(hipMallocHost(&src, BUFF_SIZE) == hipSuccess);
    assert(hipMemset(src, 0, BUFF_SIZE) == hipSuccess);
    assert(hipMemset(dst, 0, BUFF_SIZE) == hipSuccess);

    for (int i = 0; i < 10; i++)
        test_copy_one("P2P", dst, src);

    hipFreeHost(dst);
    hipFreeHost(src);
}

void test_hh(void)
{
    void *dst = malloc(BUFF_SIZE);
    assert(dst);

    void *src = malloc(BUFF_SIZE);
    assert(src);

    memset(src, 0, BUFF_SIZE);
    memset(dst, 0, BUFF_SIZE);

    for (int i = 0; i < 10; i++)
        test_copy_one("H2H", dst, src);

    free(dst);
    free(src);
}

void test_ph(int p2h)
{
    void *dst, *src;

    if (p2h) {
        dst = malloc(BUFF_SIZE);
        assert(dst);

        assert(hipMallocHost(&src, BUFF_SIZE) == hipSuccess);
    }
    else {
        src = malloc(BUFF_SIZE);
        assert(src);

        assert(hipMallocHost(&dst, BUFF_SIZE) == hipSuccess);
    }

    memset(src, 0, BUFF_SIZE);
    memset(dst, 0, BUFF_SIZE);

    for (int i = 0; i < 10; i++)
        test_copy_one(p2h ? "P2H" : "H2P", dst, src);

    if (p2h) {
        free(dst);
        hipFreeHost(src);
    }
    else {
        free(src);
        hipFreeHost(dst);
    }
}

int main(void)
{
    //test_dp(0, 0);
    //test_dp(1, 0);
    test_dp(0, 1);
    test_dp(1, 1);

    test_pp();

    test_hh();

    test_ph(0);
    test_ph(1);

    test_dd(0);
    test_dd(1);

    return 0;
}
