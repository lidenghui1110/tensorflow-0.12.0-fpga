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

#include "tensorflow/core/common_runtime/fpga/fpga_bfc_allocator.h"


namespace tensorflow {

//FPGABFCAllocator::FPGABFCAllocator(int device_id, size_t total_memory)
//    : FPGABFCAllocator(device_id, total_memory, FPGAOptions()) {}

FPGABFCAllocator::FPGABFCAllocator(FPGAWrapper* device, size_t total_memory)
    : BFCAllocator(
          new FPGAMemAllocator(device),
          total_memory, false, "fpga_bfc") {}

}  // namespace tensorflow
