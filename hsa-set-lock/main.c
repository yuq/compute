#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <sys/mman.h>

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

#define BUFF_SIZE (1 << 24)

int main(int argc, char **argv)
{
	hsa_status_t status;

	assert(hsa_init() == HSA_STATUS_SUCCESS);

	hsa_agent_t agent = get_agent(0);

	void *userptr;
	userptr = mmap(NULL, BUFF_SIZE, PROT_WRITE|PROT_READ, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0);
	assert(userptr != MAP_FAILED);
	memset(userptr, 0, BUFF_SIZE);

	void *userptr_va;
	status = hsa_amd_memory_lock(userptr, BUFF_SIZE, NULL, 0, &userptr_va);
	assert(status == HSA_STATUS_SUCCESS);
//*
	void *nullptr;
	nullptr = mmap(userptr, BUFF_SIZE, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
	assert(nullptr != MAP_FAILED);

	void *userptr2;
	userptr2 = mmap(userptr, BUFF_SIZE, PROT_WRITE|PROT_READ,
			MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
	assert(userptr2 != MAP_FAILED);
	memset(userptr2, 0, BUFF_SIZE);
	assert(userptr == userptr2);
//*/
	status = hsa_amd_memory_fill(userptr_va, 0x12345678, BUFF_SIZE/sizeof(uint32_t));
	assert(status == HSA_STATUS_SUCCESS);

	bool success = true;
	for (int i = 0; i < BUFF_SIZE/sizeof(uint32_t); i++) {
		uint32_t *data = userptr;
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
