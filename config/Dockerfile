ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

USER root

ENV MUJOCO_GL=osmesa
ENV DISPLAY :0

RUN apt update
RUN apt-get install -y libgl1-mesa-glx libosmesa6

USER jovyan

RUN pip install --upgrade pip && \
        pip install gym && \
        pip install launchpad && \
        pip install dm_control>=1.0.8 && \
        pip install git+https://github.com/deepmind/acme.git && \
        pip install imageio-ffmpeg && \
        pip install gdown && \
        pip install torch && \
        pip install tensorboard &&\
        pip install rsatoolbox &&\
        pip install dm_env

# Override command to disable running jupyter notebook at launch
CMD ["/bin/bash"]