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
REGISTER_OP("VecAddOpenCl")
    .Attr("T: {int32, float, double}")
    .Input("a: T")
    .Input("b: T")
    .Output("c: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
	
template <typename T>
class VecAddOpenClOp : public OpKernel {
public:
    explicit VecAddOpenClOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

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
		
		cl_platform_id platform_id = NULL;
		cl_device_id device_id = NULL;
		cl_context context = NULL;
		cl_command_queue command_queue = NULL;
		cl_mem cl_buf1 = NULL;
		cl_mem cl_buf2 = NULL;
		cl_program program = NULL;
		cl_kernel kernel = NULL;
		cl_uint ret_num_devices;
		cl_uint ret_num_platforms;
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
		
		/* Get platform/device information */
		ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
		ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
		
		/* Create OpenCL Context */
		context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
		
		/* Create Command Queue */
		command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
		
		/* Create memory buffer*/
		cl_buf1 = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(T), NULL, &ret);
		cl_buf2 = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(T), NULL, &ret);
		
		/* Transfer data to memory buffer */
		ret = clEnqueueWriteBuffer(command_queue, cl_buf1, CL_TRUE, 0, MEM_SIZE * sizeof(T), buf1, 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, cl_buf2, CL_TRUE, 0, MEM_SIZE * sizeof(T), buf2, 0, NULL, NULL);
		
		cl_mem buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MEM_SIZE * sizeof(cl_float), NULL, NULL );
		
		/* Create kernel program from the kernel binary */
		program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&binary_size, (const unsigned char **)&binary_buf, &binary_status, &ret);
		
		ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
		/* Create OpenCL Kernel */
		kernel = clCreateKernel(program, "vecadd", &ret);
		/* Set OpenCL kernel argument */
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&cl_buf1);
		ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&cl_buf2);
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&buffer);
		
		size_t global_work_size[3] = {MEM_SIZE, 0, 0};
		size_t local_work_size[3] = {MEM_SIZE, 0, 0};
		
		/* Execute OpenCL kernel */
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		//printf("%d\n", CL_SUCCESS);
		//printf("cal result : %d\n", ret);
		
		/* Transfer result from the memory buffer */
		ret = clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, MEM_SIZE * sizeof(T), buf, 0, NULL, NULL);
		
		/* Display result */
		// for(i=0; i<MEM_SIZE; i++){
			// printf("%f %f %f\n", buf1[i], buf2[i], buf[i]);
		// }
		// if(!memcmp(buf, cl_buf, MEM_SIZE))
			// printf("Verify passed\n");
		// else printf("verify failed\n");
		
		/* Finalization */
		ret = clFlush(command_queue);
		ret = clFinish(command_queue);
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(cl_buf1);
		ret = clReleaseMemObject(cl_buf2);
		ret = clReleaseMemObject(buffer);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);
		
		free(binary_buf);
		
    }
};

#define REGISTER_KERNEL(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
        Name("VecAddOpenCl").Device(DEVICE_FPGA).TypeConstraint<type>("T"), \
        VecAddOpenClOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL


