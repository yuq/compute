#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <time.h>

#define BUFF_SIZE (1 << 24)

void test(void *a, void *b)
{
    struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    memcpy(a, b, BUFF_SIZE);

    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE / (end - start);
	printf("copy rate %f GB/s\n", rate);
}

int main(void)
{
    void *a = malloc(BUFF_SIZE);
    assert(a);

    void *b = malloc(BUFF_SIZE);
    assert(b);

    for (int i = 0; i < 10; i++)
        test(a, b);

    free(a);
    free(b);
    return 0;
}
