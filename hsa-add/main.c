#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <unistd.h>
#include <fcntl.h>

#include <hsa.h>
#include <hsa_ext_amd.h>

static hsa_status_t test_hsa_agent_callback(hsa_agent_t agent, void* data)
{
	hsa_device_type_t device_type;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) == HSA_STATUS_SUCCESS);
	if (device_type != HSA_DEVICE_TYPE_GPU)
		return HSA_STATUS_SUCCESS;

	uint32_t features = 0;
	assert(hsa_agent_get_info(agent, HSA_AGENT_INFO_FEATURE, &features) == HSA_STATUS_SUCCESS);
	if (!(features & HSA_AGENT_FEATURE_KERNEL_DISPATCH))
		return HSA_STATUS_SUCCESS;

	hsa_agent_t *ret = data;
	*ret = agent;
	printf("found gpu agent %lu\n", agent.handle);
	return HSA_STATUS_INFO_BREAK;
}

static void test_error_callback(hsa_status_t status, hsa_queue_t *queue, void *data)
{
	const char* message;

	hsa_status_string(status, &message);

	printf("Error at queue %lu: %s", queue->id, message);
}


struct test_regions {
	hsa_region_t gtt;
	hsa_region_t vis_vram;
	hsa_region_t invis_vram;
	hsa_region_t kernarg;
};

static void init_executable(hsa_agent_t *agent, hsa_kernel_dispatch_packet_t *packet)
{
	int fd = open("hello.co", O_RDONLY);
	assert(fd >= 0);

	hsa_code_object_reader_t reader;
	assert(hsa_code_object_reader_create_from_file(fd, &reader) == HSA_STATUS_SUCCESS);

	hsa_executable_t executable;
	assert(hsa_executable_create_alt(
		       HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
		       NULL, &executable) == HSA_STATUS_SUCCESS);

	assert(hsa_executable_load_agent_code_object(
		       executable, *agent, reader, NULL, NULL) == HSA_STATUS_SUCCESS);

	assert(hsa_executable_freeze(executable, NULL) == HSA_STATUS_SUCCESS);

	//uint32_t result;
	//assert(hsa_executable_validate(executable, &result) == HSA_STATUS_SUCCESS)

	hsa_executable_symbol_t symbol;
	assert(hsa_executable_get_symbol_by_name(
		       executable, "hello", agent, &symbol) == HSA_STATUS_SUCCESS);

	assert(hsa_executable_symbol_get_info(
		       symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
		       &packet->kernel_object) == HSA_STATUS_SUCCESS);

	assert(hsa_executable_symbol_get_info(
		       symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
		       &packet->group_segment_size) == HSA_STATUS_SUCCESS);
	
	assert(hsa_code_object_reader_destroy(reader) == HSA_STATUS_SUCCESS);
	//assert(hsa_executable_destroy(executable) == HSA_STATUS_SUCCESS);
}

#define MAT_DIM_X 256
#define MAT_DIM_Y 1
#define MAT_SIZE  (MAT_DIM_X * MAT_DIM_Y * sizeof(float))

static float *in_A, *in_B, *out_C;

static void alloc_memory(hsa_agent_t *agent, struct test_regions *regions)
{
    assert(hsa_memory_allocate(regions->vis_vram, MAT_SIZE, (void **)&in_A) == HSA_STATUS_SUCCESS);
    assert(hsa_memory_allocate(regions->vis_vram, MAT_SIZE, (void **)&in_B) == HSA_STATUS_SUCCESS);
    assert(hsa_memory_allocate(regions->vis_vram, MAT_SIZE, (void **)&out_C) == HSA_STATUS_SUCCESS);
    memset(out_C, 0, MAT_SIZE);
    for (int i = 0; i < MAT_DIM_Y; i++) {
        for (int j = 0; j < MAT_DIM_X; j++) {
            in_A[i * MAT_DIM_X + j] = 1;
            in_B[i * MAT_DIM_X + j] = 5;
        }
    }
}

static void init_packet(hsa_agent_t *agent, hsa_kernel_dispatch_packet_t *packet, struct test_regions *regions)
{
	// Reserved fields, private and group memory, and completion signal are all set to 0.
	memset(((uint8_t*) packet) + 4, 0, sizeof(hsa_kernel_dispatch_packet_t) - 4);

	packet->workgroup_size_x = 256;
	packet->workgroup_size_y = 1;
	packet->workgroup_size_z = 1;
	packet->grid_size_x = 256;
	packet->grid_size_y = 1;
	packet->grid_size_z = 1;

	struct test_args {
        float *in_A;
        float *in_B;
		float *out_C;
	} *args = NULL;
	assert(hsa_memory_allocate(regions->kernarg, sizeof(args), (void **)&args) == HSA_STATUS_SUCCESS);
	packet->kernarg_address = args;

    alloc_memory(agent, regions);

    args->in_A = in_A;
    args->in_B = in_B;
    args->out_C = out_C;

	init_executable(agent, packet);

	uint16_t header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
	header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;

	uint16_t rest = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

	__atomic_store_n((uint32_t *)packet, header | (rest << 16), __ATOMIC_RELEASE);
}

hsa_status_t get_kernarg(hsa_region_t region, void* data) {

	hsa_region_segment_t segment;
	hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment);
	if (segment != HSA_REGION_SEGMENT_GLOBAL)
		return HSA_STATUS_SUCCESS;

	hsa_region_global_flag_t flags;
	hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);

	bool host_accessible_region = false;
	hsa_region_get_info(region, HSA_AMD_REGION_INFO_HOST_ACCESSIBLE, &host_accessible_region);

	struct test_regions *tr = data;
	if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
		printf("get GTT region\n");
		tr->gtt = region;
	}

	if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
		if (host_accessible_region) {
			printf("get vis VRAM region\n");
			tr->vis_vram = region;
		}
		else {
			printf("get invis VRAM region\n");
			tr->invis_vram = region;
		}
	}

	if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
		printf("get kernarg region\n");
		tr->kernarg = region;
	}

	return HSA_STATUS_SUCCESS;
}

int main(int argc, char **argv)
{
	assert(hsa_init() == HSA_STATUS_SUCCESS);

	hsa_agent_t gpu_agent;
	hsa_status_t status;
	status = hsa_iterate_agents(test_hsa_agent_callback, &gpu_agent);
	assert(status == HSA_STATUS_SUCCESS || status == HSA_STATUS_INFO_BREAK);

	struct test_regions regions;
	hsa_agent_iterate_regions(gpu_agent, get_kernarg, &regions);

	// Create a queue in the kernel agent. The queue can hold 4 packets
	hsa_queue_t *queue;
	hsa_queue_create(gpu_agent, 4, HSA_QUEUE_TYPE_SINGLE, test_error_callback, NULL,
			 UINT32_MAX, UINT32_MAX, &queue);

	// Atomically request a new packet ID.
	uint64_t packet_id = hsa_queue_load_write_index_relaxed(queue);
	uint64_t next_packet_id = packet_id + 1;

	// Wait until the queue is not full before writing the packet
	while (next_packet_id - hsa_queue_load_read_index_scacquire(queue) >= queue->size);

	// Calculate the virtual address where to place the packet
	hsa_kernel_dispatch_packet_t *packet = (hsa_kernel_dispatch_packet_t*)queue->base_address + packet_id;

	init_packet(&gpu_agent, packet, &regions);

	// Create a signal with an initial value of one to monitor the task completion
	hsa_signal_create(1, 0, NULL, &packet->completion_signal);

	// Increase queue write index
	hsa_queue_store_write_index_relaxed(queue, next_packet_id);

	// Notify the queue that the packet is ready to be processed
	hsa_signal_store_screlease(queue->doorbell_signal, packet_id);

	// Wait for the task to finish, which is the same as waiting for the value of the completion signal to be zero
	while (hsa_signal_wait_scacquire(packet->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_ACTIVE) != 0);

	printf("result %f %f %f %f\n", out_C[0], out_C[1], out_C[2], out_C[3]);

	// Done! The kernel has completed. Time to cleanup resources and leave
	hsa_signal_destroy(packet->completion_signal);
	hsa_queue_destroy(queue);
	hsa_shut_down();
	return 0;
}
