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

#define BUFF_SIZE (1 << 22)

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

void copy_one(void *dst, void *src, const char *name)
{
	struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));

	if (hsa_memory_copy(dst, src, BUFF_SIZE) != HSA_STATUS_SUCCESS) {
		printf("%s copy fail\n", name);
		return;
	}

	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

	double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE / (end - start);
	printf("%s copy rate %f GB/s\n", name, rate);
}

int main(int argc, char **argv)
{
	hsa_status_t status;

	assert(hsa_init() == HSA_STATUS_SUCCESS);

	hsa_agent_t agent = get_agent(0);
	hsa_region_t gtt_region = get_gtt_region(agent);
	hsa_region_t vram_region = get_vram_region(agent);

	void *gtt_dst;
	status = hsa_memory_allocate(gtt_region, BUFF_SIZE, &gtt_dst);
	assert(status == HSA_STATUS_SUCCESS);
	print_info(gtt_dst, "gtt dst");

	void *gtt_src;
	status = hsa_memory_allocate(gtt_region, BUFF_SIZE, &gtt_src);
	assert(status == HSA_STATUS_SUCCESS);
	print_info(gtt_src, "gtt src");
	for (int i = 0; i < BUFF_SIZE / sizeof(int); i++) {
		int *data = gtt_src;
		data[i] = i;
	}

	void *vram_dst;
	status = hsa_memory_allocate(vram_region, BUFF_SIZE, &vram_dst);
	assert(status == HSA_STATUS_SUCCESS);
	print_info(vram_dst, "vram dst");

	void *vram_src;
	status = hsa_memory_allocate(vram_region, BUFF_SIZE, &vram_src);
	assert(status == HSA_STATUS_SUCCESS);
	print_info(vram_src, "vram src");

	void *host_dst = NULL;
	assert(!posix_memalign(&host_dst, 0x200000, BUFF_SIZE));
	assert(host_dst);
	print_info(host_dst, "host dst");

	void *host_src = NULL;
	assert(!posix_memalign(&host_src, 0x200000, BUFF_SIZE));
	assert(host_src);
	print_info(host_src, "host src");

	for (int i = 0; i < 10; i++) {
		memset(gtt_dst, 0, BUFF_SIZE);
		copy_one(gtt_dst, gtt_src, "P2P");
		if (memcmp(gtt_dst, gtt_src, BUFF_SIZE))
			printf("P2P copy content is wrong\n");
		else
			printf("P2P copy content is right\n");
	}

	for (int i = 0; i < 10; i++)
		copy_one(gtt_dst, vram_src, "D2P");
	for (int i = 0; i < 10; i++)
		copy_one(vram_dst, gtt_src, "P2D");
	for (int i = 0; i < 10; i++)
		copy_one(vram_dst, vram_src, "D2D");

	for (int i = 0; i < 10; i++)
		copy_one(host_dst, host_src, "H2H");
	for (int i = 0; i < 10; i++)
		copy_one(host_dst, gtt_src, "P2H");
	for (int i = 0; i < 10; i++)
		copy_one(gtt_dst, host_src, "H2P");

	for (int i = 0; i < 10; i++)
		copy_one(host_dst, vram_src, "D2H");
	for (int i = 0; i < 10; i++)
		copy_one(vram_dst, host_src, "H2D");

	hsa_shut_down();
	return 0;
}
