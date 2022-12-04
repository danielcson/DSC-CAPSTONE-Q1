# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
# See https://github.com/ucsd-ets/datahub-docker-stack/wiki/Stable-Tag 
# for a list of the most current containers we maintain
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

ENV MUJOCO_GL=osmesa
ENV DISPLAY :0

RUN apt update
RUN apt-get install -y libgl1-mesa-glx libosmesa6

# dsmlp only uses python3.7
# RUN sudo apt-get install libpython3.9-dev

# 3) install packages using notebook user
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