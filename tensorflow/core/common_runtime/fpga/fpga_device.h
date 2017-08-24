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

//#if !TENSORFLOW_USE_FPGA
//#error This file must only be included when building TensorFlow with FPGA support
//#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_DEVICE_H_

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/fpga/fpga_device_context.h"
#include "tensorflow/core/common_runtime/fpga/fpga_bfc_allocator.h"
//#include "tensorflow/core/common_runtime/fpga/fpga_wrapper.h"
#include "tensorflow/fpga_executor/fpga_wrapper.h"
#include "tensorflow/core/public/session_options.h"


namespace tensorflow {

// CPU device implementation.
class FPGADevice : public LocalDevice {
 public:
  FPGADevice(const SessionOptions& options, const string& name,
                   size_t memory_limit, const DeviceLocality& locality,
                   const string &physical_device_desc,
                   int device_id,
                   Allocator* cpu_allocator);
  ~FPGADevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  
  Status FillContextMap(const Graph* graph,
						  DeviceContextMap* device_context_map);
  Status Sync() override { return Status::OK(); }

  static string GetShortDeviceDescription(){
	return strings::StrCat("device: 0, name FPGA, PCI bus id: 0");
  }

 private:
  Allocator* cpu_allocator_;  // Not owned
  FPGAWrapper* fpga_device_;
  FPGABFCAllocator* fpga_allocator_;
  FPGADeviceContext* device_context_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_DEVICE_H_
