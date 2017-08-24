licenses(["restricted"])

# load("//tensorflow:tensorflow.bzl", "if_cuda")

cc_library(
    name = "fpga_executor",
    srcs = glob(
        [
            "fpga_wrapper.cc",
        ],
    ),
    hdrs = [
        "fpga_wrapper.h",
    ],
    linkopts = ["-lOpenCL"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
)
