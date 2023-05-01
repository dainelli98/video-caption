FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV container docker
ENV WORKDIR="/workspaces/video_caption"

USER root

RUN echo "Install dependencies..."\
    && apt-get update\
    && apt-get install\
        unixodbc-dev\
        build-essential\
        ca-certificates\
        curl\
        git-all\
        jq\
        htop\
        less\
        nano\
        vim\
        wget\
        zsh\
        -y --no-install-recommends\
    && apt-get upgrade -y \
    && apt-get full-upgrade -y \
    && apt-get autoremove -y \
    && apt-get autoclean -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip then install poetry and wheel
RUN pip3 install --no-cache-dir --upgrade pip && pip3 install --no-cache-dir wheel poetry==1.4.0

RUN git config --global --add safe.directory "$WORKDIR"
RUN git config --global core.autocrlf false
RUN git config --global pull.rebase false
RUN git config --global core.ignoreCase false
RUN git config --global http.sslVerify false

RUN git config --global credential.helper store

RUN poetry config virtualenvs.create false

RUN poetry install --all-extras

# Install oh-my-zsh
RUN yes Y | sh -c "$(curl -k -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
