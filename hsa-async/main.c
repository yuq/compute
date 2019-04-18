#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

#define MAX_GPU 16
struct agents {
	hsa_agent_t agents[MAX_GPU];
	int num;
};

static hsa_status_t test_hsa_agent_callback(hsa_agent_t agent, void* data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	if (device_type != HSA_DEVICE_TYPE_GPU) {
		printf("found cpu agent %lu\n", agent.handle);
		return HSA_STATUS_SUCCESS;
	}

	struct agents *agents = data;
	assert(agents->num < MAX_GPU);

	agents->agents[agents->num++] = agent;
	printf("found gpu agent %lu\n", agent.handle);
	return HSA_STATUS_SUCCESS;
}

static hsa_agent_t get_agent(int idx)
{
	static struct agents agents = {0};

	if (agents.num == 0) {
		hsa_status_t status = hsa_iterate_agents(test_hsa_agent_callback, &agents);
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

static hsa_status_t test_gtt_region_callback(hsa_region_t region, void* data)
{
	hsa_region_segment_t segment;
	hsa_region_get_info(
		region, HSA_REGION_INFO_SEGMENT, &segment);
	if (HSA_REGION_SEGMENT_GLOBAL != segment)
		return HSA_STATUS_SUCCESS;

	hsa_region_global_flag_t flags = 0;
	hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

	// GTT
	if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
		hsa_region_t *ret = data;
		*ret = region;
		return HSA_STATUS_INFO_BREAK;
	}

	return HSA_STATUS_SUCCESS;
}

static hsa_region_t get_gtt_region(hsa_agent_t agent)
{
	hsa_region_t ret;
	hsa_status_t status =
		hsa_agent_iterate_regions(
			agent, test_gtt_region_callback, &ret);
	assert(status == HSA_STATUS_INFO_BREAK);
	return ret;
}

#define BUFF_NUM  16
#define BUFF_SIZE (1 << 24)

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

    void *vram_ptr[BUFF_NUM];
    for (int i = 0; i < BUFF_NUM; i++) {
        status = hsa_memory_allocate(vram_region, BUFF_SIZE, vram_ptr + i);
        assert(status == HSA_STATUS_SUCCESS);
        print_info(vram_ptr[i], "vram");
    }

	hsa_agent_t agent1 = get_agent(1);
	hsa_region_t gtt_region = get_gtt_region(agent1);

    void *gtt_ptr[BUFF_NUM];
    for (int i = 0; i < BUFF_NUM; i++) {
        status = hsa_memory_allocate(gtt_region, BUFF_SIZE, gtt_ptr + i);
        assert(status == HSA_STATUS_SUCCESS);
        print_info(gtt_ptr[i], "gtt");
    }

    hsa_signal_t signal[BUFF_NUM];
    for (int i = 0; i < BUFF_NUM; i++) {
        status = hsa_signal_create(1, 0, NULL, signal + i);
        assert(status == HSA_STATUS_SUCCESS);
    }

    struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

    for (int i = 0; i < BUFF_NUM; i++) {
        status = hsa_amd_memory_async_copy(gtt_ptr[0], agent1,
                                           vram_ptr[0], agent0,
                                           BUFF_SIZE, 0, NULL,
                                           signal[i]);
        assert(status == HSA_STATUS_SUCCESS);
    }

    for (int i = 0; i < BUFF_NUM; i++) {
        /*
        while (hsa_signal_wait_acquire(signal[i], HSA_SIGNAL_CONDITION_LT, 1,
                                       UINT64_MAX, HSA_WAIT_STATE_ACTIVE));
        */
        while (1) {
            hsa_signal_value_t v = hsa_signal_load_scacquire(signal[i]);
            printf("v=%ld\n", v);
        }
        //while (hsa_signal_load_scacquire(signal[i]) != 0);
    }

    assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

    double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE * BUFF_NUM / (end - start);
	printf("copy rate %f GB/s\n", rate);

	hsa_shut_down();
    return 0;
}
