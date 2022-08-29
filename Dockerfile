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
ARG YCM_VERSION=v0.14.0
RUN cd $CODE_DIR && \
    git clone --depth 1 --branch $YCM_VERSION https://github.com/robotology/ycm.git && \
    cd ycm && \
    mkdir build && cd build && \
    cmake .. && \
    make -j `nproc` install


# YARP
ARG YARP_VERSION=v3.4.0
RUN cd $CODE_DIR && \
    git clone --depth 1 --branch $YARP_VERSION https://github.com/robotology/yarp.git &&\
    cd yarp &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc` install

EXPOSE 10000/tcp 10000/udp
RUN yarp check

# SUPERIMPOSEMESH
ARG SIML_VERSION=devel
RUN cd $CODE_DIR &&\
    git clone --depth 1 --branch $SIML_VERSION https://github.com/robotology/superimpose-mesh-lib.git &&\
    cd superimpose-mesh-lib &&\
    mkdir build && cd build &&\
    cmake .. &&\
    make -j `nproc` install

# event-driven

ARG ED_VERSION=ev2-dev
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

ARG ssh_prv_key
ARG ssh_pub_key

RUN apt install -y \
    openssh-client git \
    libmysqlclient-dev \
    libsm6 libxext6

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh
RUN ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
    echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
    chmod 600 /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519.pub

RUN cd $CODE_DIR &&\
    git clone git@github.com:arrenglover/object-track-6dof.git