/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//#if TENSORFLOW_USE_FPGA

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include "tensorflow/core/common_runtime/fpga/fpga_device_context.h"

namespace tensorflow {

void FPGADeviceContext::CopyCPUTensorToDevice(const Tensor* cpu_tensor,
                                             Device* device,
                                             Tensor* device_tensor,
                                             StatusCallback done) const {
  const int64 total_bytes = cpu_tensor->TotalBytes();
  if(total_bytes > 0) {
	const void *src_ptr = DMAHelper::base(cpu_tensor);
  	void *dst_ptr = DMAHelper::base(device_tensor);
  	device->fpga_wrapper_device()->memcpyHostToDevice(dst_ptr, src_ptr, total_bytes);
	done(Status::OK());
  }
}

void FPGADeviceContext::CopyDeviceTensorToCPU(const Tensor* device_tensor,
                                             StringPiece tensor_name,
                                             Device* device, Tensor* cpu_tensor,
                                             StatusCallback done) {
  const int64 total_bytes = device_tensor->TotalBytes();
  if(total_bytes > 0){
	const void *src_ptr = DMAHelper::base(device_tensor);
  	void *dst_ptr = DMAHelper::base(cpu_tensor);
  	device->fpga_wrapper_device()->memcpyDeviceToHost(dst_ptr, src_ptr, total_bytes);
	done(Status::OK());
  }
}

}  // namespace tensorflow
//#endif //TENSORFLOW_USE_FPGA