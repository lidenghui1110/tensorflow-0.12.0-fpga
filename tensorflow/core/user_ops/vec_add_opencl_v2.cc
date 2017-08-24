#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/op.h"
//#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/kernels/fill_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/tensor_types.h"
//#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

#include<iostream>
#include <CL/cl.h>
#include <string.h>	

#define MAX_BINARY_SIZE (0x100000)

using namespace std;
using namespace tensorflow;
REGISTER_OP("VecAddOpenClV2")
    .Attr("T: {int32, float, double}")
    .Input("a: T")
    .Input("b: T")
    .Output("c: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
	
template <typename T>
class VecAddOpenClV2Op : public OpKernel {
public:
    explicit VecAddOpenClV2Op(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
		DCHECK_EQ(2, ctx->num_inputs());
		
        const Tensor& a = ctx->input(0);
        const Tensor& b = ctx->input(1);
		
		const TensorShape& a_shape = a.shape();
		const TensorShape& b_shape = b.shape();
		
		DCHECK_EQ(a_shape.dims(), 1);
		DCHECK_EQ(b_shape.dims(), 1);

        //auto input0 = a.vec<T>();
        //auto input1 = b.vec<T>();

        Tensor* c = nullptr;
        TensorShape output_shape;
		output_shape.AddDim(a_shape.dim_size(0));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &c));

        //auto output = c->vec<T>();

        // for(int i=0; i<output_shape.dim_size(0); i++){
			// output(i) = input0(i)+input1(i);
        // } 
		cl_platform_id platform_id = ctx->fpga_device().CLEnv->platform_id;
		cl_device_id device_id = ctx->fpga_device().CLEnv->device_id;
		cl_context context = ctx->fpga_device().CLEnv->context;
		cl_command_queue command_queue = ctx->fpga_device().CLEnv->command_queue;
		void * host_base = ctx->fpga_device().host_base;
		cl_mem device_base = ctx->fpga_device().CLEnv->device_base;
		
		cl_program program = NULL;
		cl_kernel kernel = NULL;
		
		cl_int ret;
		
		size_t MEM_SIZE = output_shape.dim_size(0);
		
		FILE *fp;
		char fileName[] = "/home/lidenghui/newdisk/supportFPGA/opencl/add.clbin";
		size_t binary_size;
		char *binary_buf;
		cl_int binary_status;
		//cl_int i;
		
		/* Load kernel source code */
		fp = fopen(fileName, "r");
		if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);	
		}
		binary_buf = (char *)malloc(MAX_BINARY_SIZE);
		binary_size = fread(binary_buf, 1, MAX_BINARY_SIZE, fp);
		fclose(fp);
		
		/*Initialize Data */
		//在host内存中创建三个缓冲区
		auto buf1 = a.template flat<T>().data();
		auto buf2 = b.template flat<T>().data();
		auto buf = c->template flat<T>().data();
		cl_buffer_region offset_a;
		offset_a.origin = (char*)buf1-(char*)host_base;
		offset_a.size = MEM_SIZE*sizeof(T);
		
		cl_buffer_region offset_b;
		offset_b.origin = (char*)buf2-(char*)host_base;
		offset_b.size = MEM_SIZE*sizeof(T);
		
		cl_buffer_region offset_c;
		offset_c.origin = (char*)buf-(char*)host_base;
		offset_c.size = MEM_SIZE*sizeof(T);
		cl_mem cl_buf1 = clCreateSubBuffer(device_base, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &offset_a, &ret);
		cl_mem cl_buf2 = clCreateSubBuffer(device_base, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &offset_b, &ret);
		cl_mem res = clCreateSubBuffer(device_base, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &offset_c, &ret);
		/* Create kernel program from the kernel binary */
		program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&binary_size, (const unsigned char **)&binary_buf, &binary_status, &ret);
		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		/* Create OpenCL Kernel */
		kernel = clCreateKernel(program, "vecadd", &ret);
		/* Set OpenCL kernel argument */
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_buf1);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_buf2);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&res);
		size_t global_work_size[3] = {MEM_SIZE, 0, 0};
		size_t local_work_size[3] = {MEM_SIZE, 0, 0};
		
		/* Execute OpenCL kernel */
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		//printf("%d\n", CL_SUCCESS);
		//printf("cal result : %d\n", ret);
		
		/* Transfer result from the memory buffer */
		//ret = clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, MEM_SIZE * sizeof(T), buf, 0, NULL, NULL);
		
		/* Display result */
		// for(i=0; i<MEM_SIZE; i++){
			// printf("%f %f %f\n", buf1[i], buf2[i], buf[i]);
		// }
		// if(!memcmp(buf, cl_buf, MEM_SIZE))
			// printf("Verify passed\n");
		// else printf("verify failed\n");
		
		/* Finalization */
		//ret = clFlush(command_queue);
		//ret = clFinish(command_queue);
		//ret = clReleaseKernel(kernel);
		//ret = clReleaseProgram(program);
		//ret = clReleaseMemObject(cl_buf1);
		//ret = clReleaseMemObject(cl_buf2);
		//ret = clReleaseMemObject(buffer);
		//ret = clReleaseCommandQueue(command_queue);
		//ret = clReleaseContext(context);
		
		free(binary_buf);
		
    }
};

#define REGISTER_KERNEL(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
        Name("VecAddOpenClV2").Device(DEVICE_FPGA).TypeConstraint<type>("T"), \
        VecAddOpenClV2Op<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL


