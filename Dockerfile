FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /code

COPY requirements.txt .
RUN apt-get update
RUN apt-get -y install sudo
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
RUN apt-get -y install gcc
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install curl unzip vim
RUN apt-get -y install git
RUN apt-get install poppler-utils -y
RUN apt-get -y install g++
RUN pip install sentencepiece
RUN pip install wandb
RUN pip install layoutparser torchvision && pip install --force-reinstall "detectron2@git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
RUN pip install awscli

RUN pip install -r requirements.txt
RUN pip install setuptools==59.5.0
RUN pip install pyopenssl --upgrade
RUN export PYTHONPATH=$PYTHONPATH:/net/nfs.cirrascale/s2-research/catherinec/layout_parsing/
RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

