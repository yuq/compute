#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

#define MAX_GPU 16
struct agents {
	hsa_agent_t agents[MAX_GPU];
	int num;
} agents = {0};

static hsa_status_t test_hsa_agent_callback(hsa_agent_t agent, void* data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	if (device_type != HSA_DEVICE_TYPE_GPU) {
		printf("found cpu agent %lu\n", agent.handle);
		return HSA_STATUS_SUCCESS;
	}

	assert(agents.num < MAX_GPU);

	agents.agents[agents.num++] = agent;
	printf("found gpu agent %lu\n", agent.handle);
	return HSA_STATUS_SUCCESS;
}

static hsa_agent_t get_agent(int idx)
{
	if (agents.num == 0) {
		hsa_status_t status = hsa_iterate_agents(test_hsa_agent_callback, NULL);
		assert(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK);
	}

	assert(idx < agents.num);
	return agents.agents[idx];
}

static hsa_status_t test_vram_region_callback(hsa_region_t region, void* data)
{
	hsa_region_segment_t segment;
	hsa_region_get_info(
		region, HSA_REGION_INFO_SEGMENT, &segment);
	if (HSA_REGION_SEGMENT_GLOBAL != segment)
		return HSA_STATUS_SUCCESS;

	hsa_region_global_flag_t flags = 0;
	hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

	// VRAM
	if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
		hsa_region_t *ret = data;
		*ret = region;
		return HSA_STATUS_INFO_BREAK;
	}

	return HSA_STATUS_SUCCESS;
}

static hsa_region_t get_vram_region(hsa_agent_t agent)
{
	hsa_region_t ret;
	hsa_status_t status =
		hsa_agent_iterate_regions(
			agent, test_vram_region_callback, &ret);
	assert(status == HSA_STATUS_INFO_BREAK);
	return ret;
}

#define BUFF_SIZE (1 << 30)

static void print_info(void *ptr, const char *name)
{
    hsa_status_t status;
	hsa_amd_pointer_info_t info = {
		.size = sizeof(hsa_amd_pointer_info_t),
	};

    status = hsa_amd_pointer_info(ptr, &info, NULL, NULL, NULL);
	assert(status == HSA_STATUS_SUCCESS);
	printf("%s type=%d agentbase=%p hostbase=%p own=%lu\n", name,
	       info.type, info.agentBaseAddress, info.hostBaseAddress,
	       info.agentOwner.handle);
}

static hsa_agent_t get_mem_agent(void *ptr)
{
    hsa_status_t status;
	hsa_amd_pointer_info_t info = {
		.size = sizeof(hsa_amd_pointer_info_t),
	};

    status = hsa_amd_pointer_info(ptr, &info, NULL, NULL, NULL);
	assert(status == HSA_STATUS_SUCCESS);
    return info.agentOwner;
}

int main(int argc, char **argv)
{
    hsa_status_t status;

	assert(hsa_init() == HSA_STATUS_SUCCESS);

    hsa_agent_t agent0 = get_agent(0);
	hsa_region_t vram_region = get_vram_region(agent0);

    void *vram_ptr;
	status = hsa_memory_allocate(vram_region, BUFF_SIZE, &vram_ptr);
	assert(status == HSA_STATUS_SUCCESS);
    print_info(vram_ptr, "vram");
    memset(vram_ptr, 0x23, BUFF_SIZE);

    void *host_ptr;
    assert(!posix_memalign(&host_ptr, 0x200000, BUFF_SIZE));
    print_info(host_ptr, "host");

    void *lock_ptr = NULL;
    status = hsa_amd_memory_lock(host_ptr, BUFF_SIZE, agents.agents, agents.num, &lock_ptr);
    assert(status == HSA_STATUS_SUCCESS);
    printf("lock ptr %p to %p\n", host_ptr, lock_ptr);

    print_info(host_ptr, "host ptr after lock");
    print_info(lock_ptr, "lock ptr");

    hsa_signal_t signal;
    status = hsa_signal_create(1, 0, NULL, &signal);
	assert(status == HSA_STATUS_SUCCESS);

    struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    status = hsa_amd_memory_async_copy(lock_ptr, get_mem_agent(lock_ptr),
                                       vram_ptr, agent0,
                                       BUFF_SIZE, 0, NULL, signal);

    assert(status == HSA_STATUS_SUCCESS);

    while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   UINT64_MAX, HSA_WAIT_STATE_ACTIVE));

	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    uint32_t *data = host_ptr;
	printf("parent result %x %x %x %x\n", data[0], data[1], data[2], data[3]);

    double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE / (end - start);
	printf("copy rate %f GB/s\n", rate);

	hsa_shut_down();
    return 0;
}
