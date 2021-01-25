#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define CL_TARGET_OPENCL_VERSION 220

#include <CL/cl.h>

static cl_platform_id get_platform(void)
{
	cl_int ret;

	cl_uint num_platforms = 0;
	ret = clGetPlatformIDs(0, NULL, &num_platforms);
	assert(ret == CL_SUCCESS);
	assert(num_platforms > 0);

	cl_platform_id *platform_ids = alloca(sizeof(cl_platform_id) * num_platforms);
	assert(platform_ids);

	ret = clGetPlatformIDs(num_platforms, platform_ids, NULL);
	assert(ret == CL_SUCCESS);

	for (int i = 0; i < num_platforms; i++) {
		char name[128];
		ret = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, sizeof(name),
					name, NULL);
		assert(ret == CL_SUCCESS);
		printf("platform %s\n", name);
	}

	return platform_ids[0];
}

static cl_device_id get_device(cl_platform_id platform_id)
{
	cl_int ret;

	cl_uint num_devices = 0;
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 0,
			     NULL, &num_devices);
	assert(ret == CL_SUCCESS);
	assert(num_devices > 0);

	cl_device_id *device_ids = alloca(sizeof(cl_device_id) * num_devices);
	assert(device_ids);

	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, num_devices, 
			     device_ids, NULL);
	assert(ret == CL_SUCCESS);

	for (int i = 0; i < num_devices; i++) {
		char name[128];
		ret = clGetDeviceInfo(device_ids[i], CL_DEVICE_NAME, sizeof(name),
				      name, NULL);
		assert(ret == CL_SUCCESS);
		printf("device %s\n", name);
	}

	return device_ids[0];
}

static cl_mem
create_and_init_buffer(cl_context context, cl_command_queue command_queue,
		       size_t size, float value)
{
	cl_int ret;

	cl_mem mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
	assert(mem_obj && ret == CL_SUCCESS);

	ret = clEnqueueFillBuffer(command_queue, mem_obj, &value, sizeof(float),
				  0, size, 0, NULL, NULL);
	assert(ret == CL_SUCCESS);

	return mem_obj;
}

static const char vec_add[] =
	"__kernel void vec_add(__global const float *A, __global const float *B,\n"
	"                      __global float *C) {\n"
	"    int i = get_global_id(0);\n"
	"    C[i] = A[i] + B[i];\n"
	"}\n";

#define VECTOR_SIZE 0x10000

int main(void)
{
	cl_int ret;

	cl_platform_id platform_id = get_platform();
	cl_device_id device_id = get_device(platform_id);

	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	assert(context && ret == CL_SUCCESS);

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(
		context, device_id, NULL, &ret);
	assert(command_queue && ret == CL_SUCCESS);

	int size = VECTOR_SIZE;
	int nb = size * sizeof(float);

	cl_mem a_mem_obj = create_and_init_buffer(context, command_queue, nb, 1.0);
	cl_mem b_mem_obj = create_and_init_buffer(context, command_queue, nb, 5.0);
	cl_mem c_mem_obj = create_and_init_buffer(context, command_queue, nb, 0);

	const char *source_str = vec_add;
	const size_t source_size = sizeof(vec_add);
	cl_program program = clCreateProgramWithSource(
		context, 1, &source_str, &source_size, &ret);
	assert(program && ret == CL_SUCCESS);

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) {
		printf("build program fail %d\n", ret);

		char buffer[2048];
		size_t len = 0;
		ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
					    sizeof(buffer), buffer, &len);
		assert(ret == CL_SUCCESS && len < 2048);
		printf("%s\n", buffer);
		return -1;
	}

	cl_kernel kernel = clCreateKernel(program, "vec_add", &ret);
	assert(kernel && ret == CL_SUCCESS);

	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	assert(ret == CL_SUCCESS);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	assert(ret == CL_SUCCESS);

	size_t global_item_size = VECTOR_SIZE;
	size_t local_item_size = 64;
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
				     &global_item_size, &local_item_size,
				     0, NULL, NULL);
	assert(ret == CL_SUCCESS);

	float *C = calloc(size, sizeof(float));
	assert(C);

	ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
				  nb, C, 0, NULL, NULL);
	assert(ret == CL_SUCCESS);

	bool success = true;
	for (int i = 0; i < size; i++) {
		if (C[i] != 6.0) {
			printf("check result fail at %d expect 6.0 but get %f\n",
			       i, C[i]);
			success = false;
			break;
		}
	}
	if (success)
		printf("success\n");
	
	clFinish(command_queue);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseMemObject(a_mem_obj);
	clReleaseMemObject(b_mem_obj);
	clReleaseMemObject(c_mem_obj);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	free(C);
	return 0;
}
