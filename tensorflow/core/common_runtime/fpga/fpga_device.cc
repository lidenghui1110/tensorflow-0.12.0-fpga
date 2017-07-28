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

#include "tensorflow/core/common_runtime/fpga/fpga_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

FPGADevice::FPGADevice(const SessionOptions& options,
                                   const string& name, size_t memory_limit,
                                   const DeviceLocality& locality,
                                   const string &physical_device_desc,
                                   int device_id,
                                   Allocator* cpu_allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_FPGA, static_cast<Bytes>(memory_limit), locality, physical_device_desc),
                  nullptr),
      cpu_allocator_(cpu_allocator),
      fpga_device_(new FPGAWrapper(device_id)),
      fpga_allocator_(new FPGABFCAllocator(fpga_device_, memory_limit)),
      device_context_(new FPGADeviceContext()){
		set_fpga_wrapper_device(fpga_device_);
}

FPGADevice::~FPGADevice() {
	device_context_->Unref();
	delete fpga_allocator_;
	delete fpga_device_;
}

void FPGADevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    op_kernel->Compute(context);
  } else {
    op_kernel->Compute(context);
  }
}

Allocator* FPGADevice::GetAllocator(AllocatorAttributes attr) {
	if(attr.on_host())
		return cpu_allocator_;
	else
		return fpga_allocator_;
}

Status FPGADevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator_, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   ProtoDebugString(tensor_proto));
  }
  Status status;
  if(alloc_attrs.on_host()){
	*tensor = parsed;
  } else {
	Tensor copy(GetAllocator(alloc_attrs), parsed.dtype(), parsed.shape());
	device_context_->CopyCPUTensorToDevice(&parsed, this, &copy, 
									[&status](const Status &s){status = s;});
	*tensor = copy;
  }
  return status;
}

Status FPGADevice::FillContextMap(const Graph* graph,
						  DeviceContextMap* device_context_map){
  device_context_map->resize(graph->num_node_ids());
  for (Node *n : graph->nodes()) {
    device_context_->Ref();
    (*device_context_map)[n->id()] = device_context_;
  }

  return Status::OK();
}

}  // namespace tensorflow
