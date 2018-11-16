#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include <hsa.h>

static hsa_status_t test_hsa_agent_callback(hsa_agent_t agent, void* data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	if (device_type == HSA_DEVICE_TYPE_GPU)
		printf("get GPU agent %lu\n", agent.handle);
	else if (device_type == HSA_DEVICE_TYPE_CPU)
		printf("get CPU agent %lu\n", agent.handle);
	else
		printf("get unknown agent %lu\n", agent.handle);

	uint32_t features = 0;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features) == HSA_STATUS_SUCCESS);
	if (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)
		printf("  is kernel agent\n");
	
	return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
	assert(hsa_init() == HSA_STATUS_SUCCESS);

	hsa_status_t status;
	status = hsa_iterate_agents(test_hsa_agent_callback, NULL);
	assert(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK);

	
}
