load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "cal_mat_mul.so",
    srcs = ["cal_mat_mul.cc"],
)

tf_custom_op_library(
    name = "zero_out.so",
    srcs = ["zero_out.cc"],
)

tf_custom_op_library(
    name = "test_fpga.so",
    srcs = ["test_fpga.cc"],
)
tf_custom_op_library(
    name = "vec_add_opencl.so",
    srcs = ["vec_add_opencl.cc"],
    linkopt = ["-lOpenCL"],
)

tf_custom_op_library(
    name = "vec_add_opencl_v2.so",
    srcs = ["vec_add_opencl_v2.cc"],
    linkopt = ["-lOpenCL"],
)

# cc_library(
#    name = "Matrix",
#    srcs = [Matrix.h],
#    linkopts = ["-lriffa"],
# )

# tf_custom_op_library(
#    name = "cal_mat_mul_fpga.so",
#    srcs = ["cal_mat_mul_fpga.cc"],
#    deps = [":Matrix"],
# )

