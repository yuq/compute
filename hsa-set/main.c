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

int main(int argc, char **argv)
{
	hsa_status_t status;

	assert(hsa_init() == HSA_STATUS_SUCCESS);

	hsa_agent_t agent = get_agent(0);
	hsa_region_t gtt_region = get_gtt_region(agent);
	hsa_region_t vram_region = get_vram_region(agent);

	void *vram_buf;
	status = hsa_memory_allocate(vram_region, BUFF_SIZE, &vram_buf);
	assert(status == HSA_STATUS_SUCCESS);

	void *gtt_buf;
	status = hsa_memory_allocate(gtt_region, BUFF_SIZE, &gtt_buf);
	assert(status == HSA_STATUS_SUCCESS);
	memset(gtt_buf, 0, BUFF_SIZE);

	status = hsa_amd_memory_fill(vram_buf, 0x12345678, BUFF_SIZE/sizeof(uint32_t));
	assert(status == HSA_STATUS_SUCCESS);

	//sleep(3);

	status = hsa_memory_copy(gtt_buf, vram_buf, BUFF_SIZE);
	assert(status == HSA_STATUS_SUCCESS);

	//sleep(3);

	bool success = true;
	for (int i = 0; i < BUFF_SIZE/sizeof(uint32_t); i++) {
		uint32_t *data = gtt_buf;
		if (data[i] != 0x12345678) {
			printf("check fail %d %x\n", i, data[i]);
			success = false;
			break;
		}
	}

	if (success)
		printf("success\n");

	hsa_shut_down();
	return 0;
}
