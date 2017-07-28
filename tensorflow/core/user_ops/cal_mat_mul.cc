#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/op.h"
//#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/kernels/fill_functor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
//#include "tensorflow/core/framework/tensor_types.h"
//#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include<iostream>
using namespace std;
using namespace tensorflow;
REGISTER_OP("CalMatMul")
    .Attr("T: {int32, float, double}")
    .Input("a: T")
    .Input("b: T")
    .Output("c: T")
	.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
		shape_inference::ShapeHandle input_0_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_0_shape));
	 
		shape_inference::ShapeHandle input_1_shape;
		TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &input_1_shape));
		
		shape_inference::DimensionHandle output_rows = c->Dim(input_0_shape, 0);
		shape_inference::DimensionHandle output_cols = c->Dim(input_1_shape, 1);
	  
		shape_inference::DimensionHandle input_0_cols = c->Dim(input_0_shape, 1);
		shape_inference::DimensionHandle input_1_rows = c->Dim(input_1_shape, 0);
		shape_inference::DimensionHandle merged;
		TF_RETURN_IF_ERROR(c->Merge(input_0_cols, input_1_rows, &merged));
	 
		c->set_output(0, c->Matrix(output_rows, output_cols));
		return Status::OK();
	});
	
template <typename T>
class CalMatMulOp : public OpKernel {
public:
    explicit CalMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
		DCHECK_EQ(2, ctx->num_inputs());
		
        const Tensor& a = ctx->input(0);
        const Tensor& b = ctx->input(1);
		
		const TensorShape& a_shape = a.shape();
		const TensorShape& b_shape = b.shape();
		
		DCHECK_EQ(a_shape.dims(), 2);
		DCHECK_EQ(b_shape.dims(), 2);
		DCHECK_EQ(a_shape.dim_size(1), b_shape.dim_size(0));

        auto input0 = a.matrix<T>();
        auto input1 = b.matrix<T>();

        Tensor* c = nullptr;
        TensorShape output_shape;
		output_shape.AddDim(a_shape.dim_size(0));
		output_shape.AddDim(b_shape.dim_size(1));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &c));

        auto output = c->matrix<T>();

        for(int i=0; i<output_shape.dim_size(0); i++){
            for(int j=0; j<output_shape.dim_size(1); j++){
                output(i,j) = 0;
                for(int k=0; k<a_shape.dim_size(1); k++){
                    output(i,j) += input0(i,k)*input1(k,j);
                }
            }
        } 
    }
};

#define REGISTER_KERNEL(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
        Name("CalMatMul").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
        CalMatMulOp<type>)

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL


