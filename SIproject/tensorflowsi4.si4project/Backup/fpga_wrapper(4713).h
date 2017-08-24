/*
This file is the wrapper of FPGA device, the basic operation including malloc memory 
on FPGA device, memory copy between CPU and FPGA is provided by Liang Kai.
A FPGA is identified by a device_id, each device matches a wrapper, and each wrapper
is owned by a FPGAdevice.
it is used at the time when a device is constructed with a device_id, and then passed to 
the allocator of the device.
*/
//#if !TENSORFLOW_USE_FPGA
//#error This file must only be included when building TensorFlow with FPGA support
//#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_WRAPPER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_WRAPPER_H_

//#include "tensorflow/core/platform/types.h"
#include<cstddef>
#include<iostream>
#include<CL/cl.h>
namespace tensorflow {

typedef long long int64;

class FPGAWrapper {	
public:	
	FPGAWrapper(int id) : device_id(id){
		CLEnv = new CLInfo();
	}
	~FPGAWrapper(){}
	void* FPGAMalloc(size_t length);
	void FPGAFree(void* ptr);
	void memcpyHostToDevice(void* dst_ptr, const void* src_ptr, int64 total_bytes) const;
	void memcpyDeviceToHost(void* dst_ptr, const void* src_ptr, int64 total_bytes) const;
private:
	int device_id;
	void* host_base;
	CLInfo* CLEnv;
};

class CLInfo {
public:
	CLInfo(){
		clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
		clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
		context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
		command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	}
	
private:
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
	cl_mem device_base;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

} // namespace tensorflow

#endif //TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_WRAPPER_H_