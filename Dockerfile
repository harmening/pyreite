FROM python:3.6-slim

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
#        libqt4-dev \
#        libphonon-dev  \
        libxml2-dev \
        libxslt1-dev \
#        libqtwebkit-dev \
        python-dev \
#        python-opengl \
#        python-qt4 \
#        python-qt4-gl \
#        python-gts \
#        python-numpy \
#        python-numpy-abi9 \
#        python3-dev \
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
#        openssl \
# openmeeg
    && apt-get install -y make \
        -y cmake \
#        python-pyside \
#        -y python2.7-dev \
#        -y python3.7-dev \
        python3-pip \
        swig \
    && pip install \
#        pyside \
        numpy \
        scipy \
#        sparse \
        h5py \
        pytest \
        numba \
#
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN cd / \
		&& GIT_URL="https://github.com/openmeeg/openmeeg/archive/30b8ee5322863b56f12ce790c8a04852a4594be0.zip" \
    && wget -P ./ $GIT_URL \
    && unzip ./30b8ee5322863b56f12ce790c8a04852a4594be0.zip -d / \
    && mv /openmeeg-30b8ee5322863b56f12ce790c8a04852a4594be0 /openmeeg \
    && rm ./30b8ee5322863b56f12ce790c8a04852a4594be0.zip 
# Correct c++ runtime error (applies only to 30b8ee5322863b56f12ce790c8a04852a4594be0)
RUN sed -i '/transmat.set(0.0);/c\        for (int i=0; i<geo.size(); i++)  {for (int j=0; j<geo.size(); j++) {transmat(i,j) = 0.0;}}' /openmeeg/OpenMEEG/src/assembleSourceMat.cpp
RUN cd /openmeeg \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_PROGRESSBAR=ON -DBLA_VENDOR=OpenBLAS -DENABLE_PYTHON=ON -DCMAKE_CXX_FLAGS="-Wno-narrowing -Wno-unused-local-typedefs -fpermissive -std=c++14" .. \
    && make \
    && make test \
    && make install \
    && cd ../.. && rm -rf cppcheck*


RUN apt-get remove -y python2.7

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


#ENTRYPOINT ["python3"]
CMD ["python", "/data/lf.py"]
