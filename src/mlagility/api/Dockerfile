# Getting image from docker hub
from httpd

# Install prerequisites
RUN apt-get update && apt-get install -y \
    build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libreadline-dev libffi-dev wget

# Download and Install Python 3.8.13
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz && \
    tar -xf Python-3.8.13.tgz && \
    cd Python-3.8.13 && \
    ./configure --enable-optimizations && \
    make altinstall

# Update Python3 and Pip3 to use 3.8.13 version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.8 1

# Set Python 3.8 as the default Python version
RUN update-alternatives --set python3 /usr/local/bin/python3.8

# Cleanup
RUN rm -rf /tmp/Python-3.8.13 && \
    rm /tmp/Python-3.8.13.tgz

RUN python3 -m pip install onnxruntime==1.15.1
