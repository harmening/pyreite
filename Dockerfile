FROM python:3.11

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
        build-essential \ 
        sudo \
        gcc \
        g++ \
        wget \
        unzip \
# openmeeg
        libopenblas-dev \
        liblapacke-dev \
        libhdf5-serial-dev \
        libmatio-dev \
        libgts-dev \
        libgts-0.7-5 \
        libc6 \
        libglib2.0-0 \
        libglib2.0-dev \
        libglu1-mesa-dev \
        libfreetype6-dev\
        libxml2-dev \
        libxslt1-dev \
        libmpfr-dev \
        libboost-dev \
        libboost-atomic-dev \
        libboost-chrono-dev \
        libboost-date-time-dev \
        libboost-system-dev \
        libboost-thread-dev \
        libeigen3-dev \
        libcgal-dev \
        libgmp-dev \
        libgmpxx4ldbl \
        libtbb-dev \
        libssl-dev \
# openmeeg
    && apt-get install -y make \
        -y cmake \
        python3-pip \
        swig \
    && pip install --upgrade pip \
        numpy \
        scipy \
        h5py \
        pytest \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN cd / \
		&& GIT_URL="https://github.com/openmeeg/openmeeg/archive/refs/tags/2.4.7.zip" \
    && wget -P ./ $GIT_URL \
    && unzip ./2.4.7.zip -d / \
    && mv /openmeeg-2.4.7 /openmeeg \
    && rm ./2.4.7.zip 
RUN cd /openmeeg \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PROGRESSBAR=ON -DBLA_VENDOR=OpenBLAS -DENABLE_PYTHON=ON -DCMAKE_CXX_FLAGS="-Wno-narrowing -Wno-unused-local-typedefs -fpermissive -std=c++14" .. \
    && make \
    && make install \
    && cd ../.. && rm -rf cppcheck*



ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/usr/local/lib/"

ADD . /

WORKDIR /

# Install pyreite
RUN python setup.py develop
# Run tests
RUN pytest tests

RUN apt-get update && apt-get install -y vim

#RUN adduser --disabled-login --gecos '' dockuser
#RUN chown dockuser:dockuser -R /tmp/
#RUN chmod +rwx /tmp/
#USER dockuser


CMD ["python", "/data/lf.py"]
