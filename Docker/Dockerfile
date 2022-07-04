FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=arg \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    locales \
    cmake \
    unzip \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    #python-gtk2 \
    #python-gobject \
    #python-tk \
    #python-pip \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    python3-dev \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##################################### PIP3 ######################################

RUN pip3 install --upgrade pip setuptools

RUN pip3 install \
    jupyter \
    jupyter_http_over_ws

RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN pip3 install \
    digi-xbee==1.3.0 \
    pyyaml \
    cython \
    catkin-tools \
    matplotlib \
    pandas \
    requests \
    scikit-image \
    scikit-learn \
    nbdev \
    open3d==0.13.0 \
    simplejson \
    pyrr \
    tensorflow \
    gdown 

RUN pip3 install \
    torch==1.5.0+cu101 \
    torchvision==0.6.0+cu101 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip3 install --upgrade opencv-python

##### Install nbgrader #####

RUN pip3 install nbgrader==0.6.1  
RUN jupyter nbextension install --system --py nbgrader --overwrite
RUN jupyter nbextension enable --system --py nbgrader
RUN jupyter serverextension enable --system --py nbgrader

##################################### setting #####################################
RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "arg ALL=(ALL)  ALL" > /etc/sudoers

# Support of nvidia-docker 2.0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

COPY ./colab_jupyter.sh /home/arg/

USER ${USER}

WORKDIR ${HOME}

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
