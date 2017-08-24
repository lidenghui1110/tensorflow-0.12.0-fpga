#!/bin/bash
set -e
set -x
sudo pip uninstall tensorflow
bazel build -c opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
mv /tmp/tensorflow_pkg/tensorflow-0.12.0-cp27-cp27m-linux_x86_64.whl /tmp/tensorflow_pkg/tensorflow-0.12.0-cp27-none-linux_x86_64.whl
sudo pip install /tmp/tensorflow_pkg/tensorflow-0.12.0-cp27-none-linux_x86_64.whl
