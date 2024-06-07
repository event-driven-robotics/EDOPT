#FROM ubuntu:20.04
FROM nvidia/opengl:1.2-glvnd-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

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
RUN apt install -y \
    vim \
    gdb

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

# event-driven

ARG ED_VERSION=master
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

RUN --mount=type=ssh cd $CODE_DIR &&\
    git clone git@github.com:event-driven-robotics/EDOPT.git &&\
    mv EDOPT object-track-6dof

# SUPERIMPOSEMESH
ARG SIML_VERSION=devel
RUN cd $CODE_DIR &&\
    git clone --depth 1 --branch $SIML_VERSION https://github.com/robotology/superimpose-mesh-lib.git &&\
    cd superimpose-mesh-lib &&\
    git apply $CODE_DIR/object-track-6dof/superimposeroi.patch &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc` install

# Build object-track-6dof
RUN cd $CODE_DIR &&\
    cd object-track-6dof/code &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc`