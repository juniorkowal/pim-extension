FROM continuumio/miniconda3

WORKDIR /app
RUN conda create --name hpim_env python=3.6 setuptools pip
SHELL ["conda", "run", "-n", "hpim_env", "/bin/bash", "-c"]

RUN pip install https://download.pytorch.org/whl/cpu/torch-1.10.2%2Bcpu-cp36-cp36m-linux_x86_64.whl#sha256=71b191eb16569d70a3d524d85ae31dd3a4a375190d8ad10a9ead515cecf7186b

COPY . /app