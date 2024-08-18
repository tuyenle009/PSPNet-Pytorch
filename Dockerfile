FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /PSPNet-VOC

COPY model ./model
COPY pascal_voc ./pascal_voc
COPY src ./src
COPY tensorboard ./tensorboard
COPY trained_model ./trained_model
COPY inference.py ./inference.py
COPY train.py ./train.py

RUN apt-get update
RUN apt-get install vim ffmpeg libsm6 libxext6 -y
# Combine pip install commands to reduce the number of layers
RUN apt install git -y
RUN pip install opencv-python
RUN pip install torchmetrics
RUN conda install tensorboard -y
RUN pip install albumentations
RUN pip install matplotlib
# Download model weights using gdown
RUN cd trained_models && gdown --id 1w5pRmLJXvmQQA5PtCbHhZc_uC4o0YbmA




