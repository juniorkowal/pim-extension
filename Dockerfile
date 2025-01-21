FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN apt-get update && apt-get install -y \
    wget \
    curl \
    bzip2 \
    python3.6 \
    python3.6-dev \
    python3-pip \
    build-essential \
    cmake \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -f /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

WORKDIR /app
COPY . /app

RUN conda create --name hpim_env python=3.6 setuptools
RUN conda activate hpim_env && pip install https://download.pytorch.org/whl/cpu/torch-1.10.2%2Bcpu-cp36-cp36m-linux_x86_64.whl#sha256=71b191eb16569d70a3d524d85ae31dd3a4a375190d8ad10a9ead515cecf7186b

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

SHELL ["conda", "run", "-n", "hpim_env", "/bin/bash", "-c"]

RUN pip install -e .
