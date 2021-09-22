FROM ubuntu:20.04

ENV TERM linux
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get upgrade -y
RUN apt-get install -y vim tmux git make build-essential csh pkg-config parallel lsb-core wget ffmpeg

RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ `lsb_release -cs` main" >> /etc/apt/sources.list
RUN echo "deb http://dk.archive.ubuntu.com/ubuntu/ `lsb_release -cs` universe" >> /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -y gcc-7 bison flex 
RUN apt-get install -y libblas3 libblas-dev 
RUN apt-get install -y libpugixml-dev 

RUN echo "deb https://build.openmodelica.org/apt `lsb_release -cs` release" | tee /etc/apt/sources.list.d/openmodelica.list
RUN apt-get install -y python3 python3-dev python3-venv python-is-python3
RUN wget -q http://build.openmodelica.org/apt/openmodelica.asc -O- | apt-key add - 
RUN apt-get update
RUN apt-get install -y openmodelica
RUN for PKG in `apt-cache search "omlib-.*" | cut -d" " -f1`; do apt-get install -y "$PKG"; done 

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get install sudo
RUN useradd -ms /bin/bash -G sudo hotgauge
USER hotgauge
WORKDIR /home/hotgauge/

COPY --chown=hotgauge:hotgauge ./ ./HotGauge

WORKDIR /home/hotgauge/HotGauge/
RUN python -m venv env
RUN source env/bin/activate && pip install -r requirements.txt
RUN ./get_and_patch_3DICE.sh

WORKDIR /home/hotgauge/HotGauge/3d-ice/
RUN ./install-superlu.sh
RUN make CC=gcc-7
RUN make plugin CC=gcc-7
RUN make test CC=gcc-7

WORKDIR /home/hotgauge/HotGauge/

USER root
RUN mkdir /data
RUN chown hotgauge:hotgauge /data
VOLUME /data
USER hotgauge

CMD ["/bin/bash"]
