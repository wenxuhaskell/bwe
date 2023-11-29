FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# this needs to avoid time zone question
ENV DEBIAN_FRONTEND=noninteractive

# install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    cmake \
    git \
    wget \
    unzip \
    python3-dev \
    zlib1g \
    zlib1g-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglu1-mesa-dev \
    libasio-dev \
    pkg-config \
    python3-tk \
    libsm6 \
    libxext6 \
    libxrender1 \
    libpcre3-dev \
    libffi-dev \
    libssl-dev


WORKDIR /root/
ENV PYTHON_VERSION=3.11.5
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

# INSTALL OK PYTHON WITH PYENV
RUN apt-get update && apt-get install -y libbz2-dev libncurses5 libncurses5-dev libncursesw5 tk-dev libreadline-dev libsqlite3-dev python3-testresources swig
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

RUN pip install --no-cache-dir \
    Cython==0.29.28 \
    git+https://github.com/takuseno/d4rl-atari && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    wget https://github.com/takuseno/d3rlpy/archive/master.zip && \
    unzip master.zip && \
    cd d3rlpy-master && \
    pip install --no-cache-dir . && \
    cd .. && \
    rm -rf d3rlpy-master master.zip

EXPOSE 6006

CMD ["tail", "-f", "/dev/null"]