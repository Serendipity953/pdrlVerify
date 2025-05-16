
FROM ubuntu:20.04


ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && apt-get install -y \
    bzip2 \
    build-essential \
    unzip \
    git \
    cmake \
    libboost-all-dev \
    libcln-dev \
    libgmp-dev \
    libginac-dev \
    automake \
    libglpk-dev \
    libhwloc-dev \
    libz3-dev \
    libxerces-c-dev \
    libeigen3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*


COPY Anaconda3-2023.03-1-Linux-x86_64.sh /app/Anaconda3-2023.03-1-Linux-x86_64.sh


RUN bash /app/Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm /app/Anaconda3-2023.03-1-Linux-x86_64.sh


ENV PATH /opt/conda/bin:$PATH








RUN conda create -n Verify python=3.7 -y && \
    conda install -n Verify pandas matplotlib pytorch -y && \
    conda install -n Verify rtree -y
RUN /opt/conda/bin/conda run -n Verify pip install portion==2.3.0 gym==0.19.0


RUN echo "source activate Verify" > ~/.bashrc
ENV CONDA_DEFAULT_ENV=Verify
ENV PATH /opt/conda/envs/Verify/bin:$PATH

COPY pdrlVerify /app/Verify

ENV PYTHONPATH=/app/Verify:$PYTHONPATH

COPY cmake-install.sh ./cmake-install.sh



RUN chmod +x cmake-install.sh && \
    ./cmake-install.sh --skip-license --prefix=/usr && \
    rm cmake-install.sh






COPY carl-storm /app/carl-storm

RUN cd /app/carl-storm && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make lib_carl


COPY storm-1.8.1.zip /app/storm-stable.zip

RUN /usr/bin/unzip /app/storm-stable -d /app && \
    rm /app/storm-stable.zip && \
    mv /app/moves-rwth-storm-294960b /storm && \
    cd /storm && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make



COPY pycarl /app/pycarl

RUN cd /app/pycarl && \
    /opt/conda/envs/Verify/bin/python setup.py develop && \
    /opt/conda/envs/Verify/bin/python setup.py test


COPY stormpy /app/stormpy

RUN cd /app/stormpy && \
    /opt/conda/envs/Verify/bin/python setup.py develop build_ext --jobs 4 develop



ENV PATH=$PATH:/storm/build/bin


WORKDIR /app

CMD ["/bin/bash", "-c", "source activate Verify && exec bash"]
