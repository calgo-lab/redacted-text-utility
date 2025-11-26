FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN pip install -U pip
RUN pip install matplotlib

RUN pip install -U tensorboard tqdm torchmetrics transformers pandas


RUN pip install accelerate
RUN pip install lightning
RUN pip install datasets
RUN pip install jsonargparse[signatures]>=4.27.7
WORKDIR /src
COPY src .
