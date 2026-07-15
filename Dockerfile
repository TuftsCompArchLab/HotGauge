# HotGauge Rocky Linux 9 environment
FROM docker.io/rockylinux/rockylinux:9

LABEL description="Rocky Linux 9 environment for building/running HotGauge"

# Avoid some interactive prompts and keep logs cleaner
ENV container=docker

# Install repo tooling, enable CRB, add EPEL, and add OpenModelica EL9 repo.
# CRB is the Rocky/RHEL 9 repo that exposes many development packages.
# OpenModelica provides an EL9 RPM repo, including openmodelica-nightly.
RUN dnf -y update && \
    dnf -y install dnf-plugins-core epel-release && \
    dnf config-manager --set-enabled crb && \
    dnf config-manager --add-repo https://build.openmodelica.org/linux/rpm/el9/omc.repo && \
    dnf clean all

# Install HotGauge system dependencies.
#
# Notes:
# - python3 on Rocky 9 is Python 3.9.
# - glibc-devel.i686, libstdc++.i686, and libgcc.i686 are included for McPAT's
#   older 32-bit build expectations.
# - ffmpeg-free-devel is included for analysis tooling.
# - git, tar, which, findutils, diffutils, procps-ng, vim, and nano are practical
#   additions for working inside the container.
RUN dnf -y install \
        python3 \
        python3-devel \
        python3-pip \
        gcc \
        gcc-c++ \
        make \
        cmake \
        bison \
        flex \
        pkg-config \
        boost-devel \
        sqlite-devel \
        zlib-devel \
        openblas-devel \
        openblas-openmp \
        bzip2-devel \
        xz \
        wget \
        unzip \
        zip \
        csh \
        parallel \
        pugixml \
        pugixml-devel \
        openmodelica-nightly \
        ffmpeg-free-devel \
        glibc-devel.i686 \
        libstdc++.i686 \
        libgcc.i686 \
        git \
        tar \
        which \
        findutils \
        diffutils \
        procps-ng \
        vim \
        patch && \
    chmod -R a+rX /opt/openmodelica-nightly/share/omc/runtime/c/fmi/buildproject/ && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Useful defaults
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

# Workspace for cloning/building HotGauge
WORKDIR /workspace

CMD ["/bin/bash"]
