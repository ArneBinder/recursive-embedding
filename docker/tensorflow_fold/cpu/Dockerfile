FROM conda/miniconda3

WORKDIR /root


###################
# Setup OpenJDK 1.8
###################
#RUN \
#  apt-get install -y software-properties-common \
#  && add-apt-repository -y ppa:openjdk-r/ppa \
#  && apt-get update \
#  && apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre-headless \
#  && apt-get install -y apt-transport-https \
#  && apt-get install -y wget \
#  && apt-get clean \
#  && rm -rf /var/lib/apt/lists/*

#ENV \
#  JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/


###################
# Setup Bazel 0.7.0
###################
RUN \
  apt-get update \
  && apt-get install -y zip curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV \
  BAZEL_VERSION=0.7.0 

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc

# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc

# Install the most recent bazel release.
RUN mkdir /root/bazel && \
    cd /root/bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    chmod a+x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    rm -f bazel-$BAZEL_VERSION-installer-linux-x86_64.sh


#####################################
# get tensorflow and tensorflow fold
#####################################
RUN \
  apt-get update \
  && apt-get install -y git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN \
 git clone --recurse-submodules https://github.com/tensorflow/fold
 #&& cd $TENSORFLOW_HOME \ 
 #&& git reset --hard 4c0052d 



######################
# build prerequisites
######################

# numpy is necessary
RUN \
  conda install --yes mkl-service numpy \
  && conda clean --all --yes

# install build-essential
RUN \
  apt-get update \
  && apt-get install -y build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


######################
# Configure TensorFlow
######################

# Need this inside Docker for nvidia-docker build step HACK
#ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
#RUN \
#  mkdir -p /usr/local/nvidia/lib64/ \
#  && cd /usr/local/nvidia/lib64/ \
#  && ln -s /usr/local/cuda-8.0/targets/x86_64-linux/lib/stubs/libcuda.so libcuda.so.1
#RUN \
#  ldconfig /usr/local/cuda/lib64


#TF_MKL_ROOT=/usr/local/bin
#ENV TF_NEED_CUDA=1
ENV TF_NEED_CUDA=0
ENV TF_NEED_GCP=0
ENV TF_NEED_JEMALLOC=1
ENV TF_NEED_HDFS=1
ENV TF_NEED_OPENCL=0
ENV TF_ENABLE_XLA=1
#ENV TF_CUDA_VERSION=8.0
#ENV TF_CUDNN_VERSION=5
#ENV CUDA_PATH="/usr/local/cuda"
#ENV CUDA_TOOLKIT_PATH=$CUDA_PATH
#ENV CUDNN_INSTALL_PATH=$CUDA_PATH
ENV PYTHON_BIN_PATH=/usr/local/bin/python3.6
ENV PYTHON_LIB_PATH=/usr/local/lib/python3.6
ENV CI_BUILD_PYTHON=$PYTHON_BIN_PATH
ENV CC_OPT_FLAGS="-march=native"
#ENV TF_CUDA_COMPUTE_CAPABILITIES=3.7
####################
###      AWS     ###
##  P2 Instances  ##
# Tesla K-80 (3.7) #
#                  #
##  G2 Instances  ##
# GRID K520 (3.5)  #
#                  #
###  Google GCP  ###
# Tesla K-80 (3.7) #
####################

# This build command is inspired by the following resources:
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/ci_build/xla/linux/gpu/run_py3.sh
#   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.devel-gpu
#   http://ci.tensorflow.org/job/tensorflow-master-linux-xla/104/consoleText
#   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/ci_build

#RUN \
#  cd $TENSORFLOW_HOME \
#  && yes "" | ./configure

ENV \
  CPU_COUNT=4
#  CPU_COUNT=8 \
#  GPU_COUNT=1 

###################
# Build TensorFlow
###################
ENV \
  TENSORFLOW_HOME=/root/fold/tensorflow

RUN \
  cd $TENSORFLOW_HOME \
  && tensorflow/tools/ci_build/builds/configured CPU \
  && sed -i.bak 's/EXPECT_FALSE(isnan(value));/EXPECT_FALSE(0);/g' tensorflow/core/kernels/mfcc_test.cc \
  && bazel build -c opt \
     --config=mkl \
#     --config=cuda \
#  && tensorflow/tools/ci_build/ci_build.sh CPU \
     --jobs=${CPU_COUNT} \
     --verbose_failures \
     --test_timeout 300,450,1200,3600 \
     --test_output=errors \
#     --local_test_jobs=${GPU_COUNT} \
#     --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.1 --copt=-msse4.2 \
     --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
#     --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute -- \
      //tensorflow/tools/pip_package:build_pip_package \
      && bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg \
      && bazel clean --expunge \
      && rm -rf /root/.cache


########################
# Build TensorFlow Fold
########################
ENV \
  TENSORFLOW_FOLD_HOME=/root/fold

RUN \
  cd $TENSORFLOW_FOLD_HOME \
  && bazel build -c opt \
     --jobs=${CPU_COUNT} \
     --verbose_failures \
     --test_timeout 300,450,1200,3600 \
     --test_output=errors \
      //tensorflow_fold/util:build_pip_package \
  && bazel-bin/tensorflow_fold/util/build_pip_package /tmp/fold_pkg \
  && bazel clean --expunge \
  && rm -rf /root/.cache


# todo: optimze (size increases by ~400mb. do directly after building)
#######################################
# Install Tensorflow & Tensorflow Fold
#######################################

RUN \
  pip --no-cache-dir install /tmp/tensorflow_pkg/tensorflow-*.whl \
  && pip --no-cache-dir install /tmp/fold_pkg/tensorflow_fold-*.whl \
  && rm -rf /tmp/tensorflow_pkg \
  && rm -rf /tmp/fold_pkg