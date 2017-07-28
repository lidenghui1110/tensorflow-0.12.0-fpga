#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("a: int32")
	.Input("b: int32")
    .Output("c: int32");

class ZeroOutOp : public OpKernel {
public:
    explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        const Tensor& input_tensor_0 = context->input(0);
		const Tensor& input_tensor_1 = context->input(1);
        auto input_0 = input_tensor_0.matrix<int32>();
		auto input_1 = input_tensor_1.matrix<int32>();

        // Create an output tensor
        Tensor* output_tensor = NULL;
		TensorShape outputshape({a.dim_size(0), b.dim_size(1)});
        OP_REQUIRES_OK(context, context->allocate_output(0, outputshape,
                                                     &output_tensor));
        auto output = output_tensor->matrix<int32>();

        // Set all but the first element of the output tensor to 0.
        const int N = input_0.size();
        for (int i = 0; i < N; i++) {
            output(i) = input_0(i)+input_1(i);
        }

        // Preserve the first input value if possible.
        //if (N > 0) output(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
