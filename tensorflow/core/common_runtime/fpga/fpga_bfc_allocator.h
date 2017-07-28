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

#ifndef TENSORFLOW_COMMON_RUNTIME_FPGA_FPGA_BFC_ALLOCATOR_H_
#define TENSORFLOW_COMMON_RUNTIME_FPGA_FPGA_BFC_ALLOCATOR_H_


#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/bfc_allocator.h"
//#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

//#include "tensorflow/core/common_runtime/fpga/fpga_wrapper.h"
#include "tensorflow/fpga_executor/fpga_wrapper.h"

//namespace gpu = ::perftools::gputools;

namespace tensorflow {

// A FPGA memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class FPGABFCAllocator : public BFCAllocator {
 public:
  FPGABFCAllocator(FPGAWrapper* device, size_t total_memory);
  //FPGABFCAllocator(int device_id, size_t total_memory,
  //                const FPGAOptions& fpga_options);
  virtual ~FPGABFCAllocator() {}

  TF_DISALLOW_COPY_AND_ASSIGN(FPGABFCAllocator);
};

// Suballocator for FPGA memory.
class FPGAMemAllocator : public SubAllocator {
 public:
  FPGAMemAllocator(FPGAWrapper* device) : device_(device){}
  ~FPGAMemAllocator() override {}

  void* Alloc(size_t alignment, size_t num_bytes) override {
    void* ptr = nullptr;
    if (num_bytes > 0) {
      ptr = device_->FPGAMalloc(num_bytes);
    }
    return ptr;
  }

  void Free(void* ptr, size_t num_bytes) override {
    if (ptr != nullptr) {
      device_->FPGAFree(ptr);
    }
  }

 private:
  FPGAWrapper *device_;
  TF_DISALLOW_COPY_AND_ASSIGN(FPGAMemAllocator);
};


}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_FPGA_FPGA_BFC_ALLOCATOR_H_
