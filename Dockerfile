FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib
ENV LIBVA_DRIVER_NAME=d3d12
ARG CODE_DIR=/usr/local/src

RUN apt update

#basic environment
RUN apt install -y \
    ca-certificates \
    build-essential \
    git \
    cmake \
    cmake-curses-gui \
    libace-dev \
    libassimp-dev \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    libeigen3-dev

# Suggested dependencies for YARP
RUN apt update && apt install -y \
    qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
    qml-module-qtquick2 qml-module-qtquick-window2 \
    qml-module-qtmultimedia qml-module-qtquick-dialogs \
    qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel \
    qml-module-qt-labs-settings \
    libqcustomplot-dev \
    libgraphviz-dev \
    libjpeg-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

RUN echo "deb [arch=amd64 trusted=yes] https://apt.prophesee.ai/dists/public/b4b3528d/ubuntu focal sdk" >> /etc/apt/sources.list &&\
    apt update

RUN apt install -y \
    libcanberra-gtk-module \
    mesa-utils \
    ffmpeg \
    libboost-program-options-dev \
    libopencv-dev \
    metavision-sdk

#my favourites
RUN apt update && apt install -y \
    vim \
    gdb

# Github CLI
RUN (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# YCM
ARG YCM_VERSION=v0.15.2
RUN cd $CODE_DIR && \
    git clone --depth 1 --branch $YCM_VERSION https://github.com/robotology/ycm.git && \
    cd ycm && \
    mkdir build && cd build && \
    cmake .. && \
    make -j `nproc` install


# YARP
ARG YARP_VERSION=v3.8.0
RUN cd $CODE_DIR && \
    git clone --depth 1 --branch $YARP_VERSION https://github.com/robotology/yarp.git &&\
    cd yarp &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc` install

EXPOSE 10000/tcp 10000/udp
RUN yarp check

# dv-processing
# Add toolchain PPA and install gcc-13/g++-13
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install -y gcc-13 g++-13

# Add inivation PPA and install dv-processing dependencies
RUN add-apt-repository ppa:inivation-ppa/inivation && \
    apt-get update && \
    apt-get install -y \
        boost-inivation \
        libcaer-dev \
        libfmt-dev \
        liblz4-dev \
        libzstd-dev \
        libssl-dev && \
    apt-get -y install dv-processing


# event-driven
ARG ED_VERSION=main
RUN cd $CODE_DIR &&\
    git clone --depth 1 --branch $ED_VERSION https://github.com/robotology/event-driven.git &&\
    cd event-driven &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc` install

# object-tracking-six-dof

######################
# set github ssh keys #
#######################

RUN apt install -y \
    openssh-client git \
    libmysqlclient-dev \
    libsm6 libxext6

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh
RUN ssh-keyscan github.com > /root/.ssh/known_hosts

ARG GIT_BRANCH=main
RUN cd $CODE_DIR &&\
    git clone https://github.com/event-driven-robotics/EDOPT.git &&\
    cd EDOPT &&\
    git checkout $GIT_BRANCH

# SUPERIMPOSEMESH
ARG SIML_VERSION=devel
RUN cd $CODE_DIR &&\
    git clone --depth 1 --branch $SIML_VERSION https://github.com/robotology/superimpose-mesh-lib.git &&\
    cd superimpose-mesh-lib &&\
    git apply $CODE_DIR/EDOPT/superimposeroi.patch &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc` install

# Build EDOPT
RUN cd $CODE_DIR &&\
    cd EDOPT/code &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc`