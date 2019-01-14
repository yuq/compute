#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <fcntl.h>

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

static void agent_init(void)
{
    hsa_status_t status = hsa_iterate_agents(test_hsa_agent_callback, NULL);
    assert(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK);
}

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

int main(int argc, char **argv)
{
	assert(hsa_init() == HSA_STATUS_SUCCESS);
    agent_init();

    void *ptr;
    int size = 0x100000;
    assert(!posix_memalign(&ptr, 0x200000, size));

    print_info(ptr, "ptr");

    hsa_status_t status;
    hsa_amd_ipc_memory_t ipc = {0};
    status = hsa_amd_ipc_memory_create(ptr, size, &ipc);
    printf("create with raw ptr %s\n",
           status == HSA_STATUS_SUCCESS ? "pass" : "fail");

    void *lock_ptr = NULL;
    status = hsa_amd_memory_lock(ptr, size, agents.agents, agents.num, &lock_ptr);
    assert(status == HSA_STATUS_SUCCESS);
    printf("lock ptr %p to %p\n", ptr, lock_ptr);

    print_info(ptr, "ptr after lock");
    print_info(lock_ptr, "lock ptr");

    status = hsa_amd_ipc_memory_create(ptr, size, &ipc);
    printf("create with raw ptr after lock %s\n",
           status == HSA_STATUS_SUCCESS ? "pass" : "fail");

    status = hsa_amd_ipc_memory_create(lock_ptr, size, &ipc);
    printf("create with locked ptr %s\n",
           status == HSA_STATUS_SUCCESS ? "pass" : "fail");

	hsa_shut_down();

	return 0;
}
