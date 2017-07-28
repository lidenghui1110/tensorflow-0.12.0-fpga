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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_DEVICE_CONTEXT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_FPGA_DEVICE_CONTEXT_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

class FPGADeviceContext : public DeviceContext {
public:
  FPGADeviceContext() {}
  ~FPGADeviceContext() override {}

  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor,
                             StatusCallback done) const override;

  void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece edge_name,
                             Device* device, Tensor* cpu_tensor,
                             StatusCallback done) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_FPGA_DEVICE_CONTEXT_H_

