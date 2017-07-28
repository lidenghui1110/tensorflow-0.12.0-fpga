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
	//return (void*)new void[length];
	return (void*)malloc(length);
}

void FPGAWrapper::FPGAFree(void* ptr) {
//	delete [] ptr;
	free(ptr);
}

void FPGAWrapper::memcpyHostToDevice(void* dst_ptr, const void* src_ptr, int64 total_bytes) const {
	memcpy(dst_ptr, src_ptr, total_bytes);
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
	memcpy(dst_ptr, src_ptr, total_bytes);
	/*
	ofstream fpga_log;
	fpga_log.open("~/newdisk/tensorflow-0.12.0-fpga/fpga_log.txt", ios::app);
	if(fpga_log.is_open()){
		fpga_log << "memory copy from FPGA to CPU : " << total_bytes <<"\n";
		fpga_log.close();
	}*/
	cout<<"==========================================="<<endl;
	cout << "memory copy from FPGA to CPU : " << total_bytes <<endl;
}

}

