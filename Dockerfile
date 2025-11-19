# Borrowed from https://github.com/openai/mujoco-py/blob/master/Dockerfile
# ------------------------------------------------------------------------
#
# (!) TODO: update base image according to the cuda version of the native machine
# Base images available at https://hub.docker.com/r/nvidia/cuda/tags
# Recommend using *cuddnnX-devel*
#
# NOTE: cuda version should match version in nvidia-smi output on native machine
 # e.g., lincoln hpc 
#FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04
# e.g., home gpu
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

# Use bourne-again shell
SHELL ["/bin/bash", "-c"]

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    zip \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3 \
    patchelf \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    tree \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update

ENV LANG C.UTF-8

# Anaconda setup
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH /opt/conda/bin:${PATH}
RUN pip install --upgrade pip
# Accept terms of service -_- also, must keep this order of packages T.T
# https://www.anaconda.com/docs/getting-started/working-with-conda/channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda install -c conda-forge libstdcxx-ng=12
RUN conda install python=3.10

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

# Lib references
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
# ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /hints
# Copy over just requirements.txt and bash scripts at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./vendor /hints/
COPY ./requirements.txt /hints/
#COPY ./setup.bash /hints/
#RUN echo "source /hints/setup.bash" >> /hints/tool_setup.bash
#RUN echo "unset LD_PRELOAD" >> /hints/tool_setup.bash
#RUN chmod +x /hints/tool_setup.bash
#RUN /hints/tool_setup.bash
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# Install cuda compatible version of torch
# https://pytorch.org/get-started/previous-versions/
# (!) TODO: update the following url, updating the cuda version to match that of the native machine
# RUN python3 -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 # lincoln hpc
RUN python3 -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 # home gpu
RUN python3 -m pip install -U gym[classic_control,mujoco,atari]
RUN python3 -m pip install -U 'mujoco-py<2.2,>=2.1'
RUN python3 -m pip install git+https://github.com/aravindr93/mjrl.git 
# Install custom sb3 for compatibility with gym 0.24+
#RUN python3 -m pip install git+https://github.com/carlosluis/stable-baselines3@fix_tests
#RUN python3 -m pip install stable-baselines3[extra]
# Compile mujoco
RUN python3 -m pip install "cython<3"
RUN python3 -c "import mujoco_py;print(mujoco_py.__version__)"

# Copy over project
COPY . /hints
RUN pip install -e .
ENTRYPOINT ["python3", "/hints/vendor/Xdummy-entrypoint"]
