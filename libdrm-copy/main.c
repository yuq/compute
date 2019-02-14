#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <xf86drm.h>
#include <amdgpu.h>
#include <amdgpu_drm.h>

#define BUFF_SIZE (1 << 24)

void *alloc_buf(amdgpu_device_handle device_handle)
{
    amdgpu_bo_handle buf_handle;
    struct amdgpu_bo_alloc_request req = {
        .alloc_size = BUFF_SIZE,
        .phys_alignment = 256,
        .preferred_heap = AMDGPU_GEM_DOMAIN_GTT,
        .flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
    };

    assert(!amdgpu_bo_alloc(device_handle, &req, &buf_handle));

    struct amdgpu_bo_info info;
    assert(!amdgpu_bo_query_info(buf_handle, &info));
    printf("heap=%d flags=%lx\n", info.preferred_heap, info.alloc_flags);

    void *cpu;
    assert(!amdgpu_bo_cpu_map(buf_handle, &cpu));
    printf("cpu addr=%p\n", cpu);

    return cpu;
}

void copy_one(void *dst, void *src)
{
    struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    memcpy(dst, src, BUFF_SIZE);

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
    int fd = open("/dev/dri/renderD128", O_RDWR);
    assert(fd);

    uint32_t major_version, minor_version;
    amdgpu_device_handle device_handle;
    assert(!amdgpu_device_initialize(fd, &major_version, &minor_version, &device_handle));

    struct amdgpu_gpu_info di;
    assert(!amdgpu_query_gpu_info(device_handle, &di));
    printf("asic id %x\n", di.asic_id);

    void *dst = alloc_buf(device_handle);
    void *src = alloc_buf(device_handle);

    for (int i = 0; i < 10; i++)
        copy_one(dst, src);

    return 0;
}
