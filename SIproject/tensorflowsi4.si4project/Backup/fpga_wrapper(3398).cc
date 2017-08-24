/*
the implementaion of fpga_wapper provided by Liang Kai
nothing to say just for beauty!
*/

//#include "tensorflow/core/common_runtime/fpga/fpga_wrapper.h"
#include "tensorflow/fpga_executor/fpga_wrapper.h"
#include <string.h>
#include <malloc.h>

using namespace std;
namespace tensorflow {

void* FPGAWrapper::FPGAMalloc(size_t length) {
	/*
	ofstream fpga_log;
	fpga_log.open("~/newdisk/tensorflow-0.12.0-fpga/fpga_log.txt", ios::app);
	if(fpga_log.is_open()){
		fpga_log << "malloc memory for FPGA : " << length <<"\n";
		fpga_log.close();
	}*/
	cout<<"==========================================="<<endl;
	cout<<"malloc memory for FPGA : " << length <<endl;
	CLEnv->device_base = clCreateBuffer(CLEnv->context, CL_MEM_READ_WRITE, length, NULL, &(CLEnv->ret));
	cout<<"==========================================="<<CLEnv->ret<<endl;
	host_base = (void*)malloc(length);
	return host_base;
}

void FPGAWrapper::FPGAFree(void* ptr) {
	clReleaseMemObject(CLEnv->device_base);
	free(ptr);
}

void FPGAWrapper::memcpyHostToDevice(void* dst_ptr, const void* src_ptr, int64 total_bytes) const {
	//memcpy(dst_ptr, src_ptr, total_bytes);
	cout<<"=========================================== memcpyHostToDevice"<<endl;
	int offset = ((char*)dst_ptr-(char*)host_base)*sizeof(void);
	CLEnv->ret = clEnqueueWriteBuffer(CLEnv->command_queue, CLEnv->device_base, CL_TRUE, offset, total_bytes, src_ptr, 0, NULL, NULL);
	if(CLEnv->ret!=0)	cout<<"memcpyHostToDevice error!"<<endl;
	/*
	ofstream fpga_log;
	fpga_log.open("~/newdisk/tensorflow-0.12.0-fpga/fpga_log.txt", ios::app);
	if(fpga_log.is_open()){
		fpga_log << "memory copy from CPU to FPGA : " << total_bytes <<"\n";
		fpga_log.close();
	}*/
	cout<<"==========================================="<<endl;
	cout << "memory copy from CPU to FPGA : " << total_bytes <<endl;
}

void FPGAWrapper::memcpyDeviceToHost(void* dst_ptr, const void* src_ptr, int64 total_bytes) const {
	//memcpy(dst_ptr, src_ptr, total_bytes);
	
	cout<<"===========================================  memcpyDeviceToHost "<<endl;
	int offset = ((char*)src_ptr-(char*)host_base)*sizeof(void);
	CLEnv->ret = clEnqueueReadBuffer(CLEnv->command_queue, CLEnv->device_base, CL_TRUE, offset, total_bytes, dst_ptr, 0, NULL, NULL);
	/*
	ofstream fpga_log;
	fpga_log.open("~/newdisk/tensorflow-0.12.0-fpga/fpga_log.txt", ios::app);
	if(fpga_log.is_open()){
		fpga_log << "memory copy from FPGA to CPU : " << total_bytes <<"\n";
		fpga_log.close();
	}*/
	if(CLEnv->ret!=0)	cout<<"memcpyHostToDevice error!"<<endl;
	cout<<"==========================================="<<endl;
	cout << "memory copy from FPGA to CPU : " << total_bytes <<endl;
}

}
