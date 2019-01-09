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
	if (device_type != HSA_DEVICE_TYPE_GPU)
		return HSA_STATUS_SUCCESS;

	struct agents *agents = data;
	assert(agents->num < MAX_GPU);

	agents->agents[agents->num++] = agent;
	printf("found gpu agent %lu\n", agent.handle);
	return HSA_STATUS_SUCCESS;
}

static hsa_agent_t get_agent(int idx)
{
	assert(hsa_init() == HSA_STATUS_SUCCESS);

	struct agents agents = {0};
	hsa_status_t status = hsa_iterate_agents(test_hsa_agent_callback, &agents);
	assert(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK);

	assert(idx < agents.num);
	return agents.agents[idx];
}

static hsa_status_t test_pool_callback(hsa_amd_memory_pool_t pool, void* data)
{
	hsa_amd_segment_t segment;
	hsa_amd_memory_pool_get_info(
		pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
	if (HSA_AMD_SEGMENT_GLOBAL != segment)
		return HSA_STATUS_SUCCESS;

	bool alloc = false;
	hsa_amd_memory_pool_get_info(
		pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
	if (!alloc)
		return HSA_STATUS_SUCCESS;

	uint32_t flags = 0;
	hsa_amd_memory_pool_get_info(
		pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);

	// VRAM
	if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
		hsa_amd_memory_pool_t *ret = data;
		*ret = pool;
		return HSA_STATUS_INFO_BREAK;
	}

	return HSA_STATUS_SUCCESS;
}

static hsa_amd_memory_pool_t get_pool(hsa_agent_t agent)
{
	hsa_amd_memory_pool_t ret;
	hsa_status_t status =
		hsa_amd_agent_iterate_memory_pools(
			agent, test_pool_callback, &ret);
	assert(status == HSA_STATUS_INFO_BREAK);
	return ret;
}

#define BUFF_SIZE (1 << 29)

static void parent(int fd)
{
	hsa_agent_t agent = get_agent(0);
	hsa_amd_memory_pool_t pool = get_pool(agent);

	hsa_status_t status;
	void *src_ptr;
	status = hsa_amd_memory_pool_allocate(pool, BUFF_SIZE, 0, &src_ptr);
	assert(status == HSA_STATUS_SUCCESS);

	hsa_amd_ipc_memory_t ipc = {0};
	assert(read(fd, &ipc, sizeof(ipc)) == sizeof(ipc));

	void *dst_ptr;
	status = hsa_amd_ipc_memory_attach(&ipc, BUFF_SIZE, 1, &agent, &dst_ptr);
	assert(status == HSA_STATUS_SUCCESS);

	hsa_signal_t signal;
	status = hsa_signal_create(1, 0, NULL, &signal);
	assert(status == HSA_STATUS_SUCCESS);

	struct timespec tv1, tv2;
	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv1));
	
	status = hsa_amd_memory_async_copy(dst_ptr, agent,
					   src_ptr, agent,
					   BUFF_SIZE, 0,
					   NULL, signal);
	assert(status == HSA_STATUS_SUCCESS);

	while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
				       (uint64_t)-1, HSA_WAIT_STATE_ACTIVE));

	assert(!clock_gettime(CLOCK_MONOTONIC_RAW, &tv2));

	double start = tv1.tv_sec;
	start = start * 1e9 + tv1.tv_nsec;

	double end = tv2.tv_sec;
	end = end * 1e9 + tv2.tv_nsec;

	double rate = BUFF_SIZE / (end - start);
	printf("copy rate %f GB/s\n", rate);

	hsa_shut_down();
}

static void child(int fd)
{
	hsa_agent_t agent = get_agent(1);
	hsa_amd_memory_pool_t pool = get_pool(agent);

	hsa_status_t status;
	void *ptr;
	status = hsa_amd_memory_pool_allocate(pool, BUFF_SIZE, 0, &ptr);
	assert(status == HSA_STATUS_SUCCESS);

	hsa_amd_ipc_memory_t ipc = {0};
	status = hsa_amd_ipc_memory_create(ptr, BUFF_SIZE, &ipc);
	assert(status == HSA_STATUS_SUCCESS);

	assert(write(fd, &ipc, sizeof(ipc)) == sizeof(ipc));

	sleep(5);
	hsa_shut_down();
}

int main(int argc, char **argv)
{
	int pipefd[2];
	assert(!pipe(pipefd));

	pid_t pid = fork();
	if (pid)
		parent(pipefd[0]);
	else
		child(pipefd[1]);

	return 0;
}
