FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY . /app/

RUN cd /app && pip3 install -r requirements.txt

LABEL maintainer=dhiab

WORKDIR /app

