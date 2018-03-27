FROM tensorflowfold_conda:pre


########################################
# get TensorFlow Fold and TensorFlow   #
########################################

# fold with tensorflow 1.1 (works):
#ARG TD_BRANCH=7c498a4   
# fold with tensorflow 1.3:
#ARG TD_BRANCH=b6ee778
ARG TD_BRANCH=af06b85 
# fold with tensorflow 1.4:
#ARG TD_BRANCH=cc9bcef

RUN echo use TD_BRANCH=${TD_BRANCH}

RUN git clone --recurse-submodules https://github.com/tensorflow/fold \
    && cd fold && git checkout ${TD_BRANCH} \
    && git submodule update --recursive

#RUN cd /fold/tensorflow && tensorflow/tools/ci_build/builds/configured CPU
# necessary to find "configure" script when executing "configured" (re-check!)
WORKDIR /fold/tensorflow
ENV TF_NEED_MKL=1
# updatedb, otherwise locate (when downloading MKL) is not working 
RUN updatedb && tensorflow/tools/ci_build/builds/configured CPU

#ARG BUILD_CMD='bazel build --verbose_failures --config=opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"'
#RUN echo BUILD_CMD=${BUILD_CMD}

ARG WHEEL_OUT=/wheel_pkg

WORKDIR /fold

RUN export PYTHON_BIN_PATH="$(which python)" \
    && bazel build --verbose_failures --config=opt --config=mkl --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow_fold/util:build_pip_package \
    && bazel-bin/tensorflow_fold/util/build_pip_package ${WHEEL_OUT}/fold_pkg \
    && cd /fold/tensorflow \
    && bazel build --verbose_failures --config=opt --config=mkl --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" //tensorflow/tools/pip_package:build_pip_package \
    && bazel-bin/tensorflow/tools/pip_package/build_pip_package ${WHEEL_OUT}/tensorflow_pkg \
    && pip --no-cache-dir install ${WHEEL_OUT}/fold_pkg/* && pip --no-cache-dir install ${WHEEL_OUT}/tensorflow_pkg/* \
    && rm -rf /tmp && rm -rf /root/.cache


WORKDIR /root


CMD [ "/bin/bash" ]

