FROM quay.io/rockylinux/rockylinux:8.10
RUN dnf install -y --setopt=install_weak_deps=False curl ca-certificates tar bzip2 gzip findutils which git && dnf clean all
ARG MICROMAMBA_VERSION=2.3.3
RUN curl --fail --location --retry 3 --max-time 180 https://micro.mamba.pm/api/micromamba/linux-64/${MICROMAMBA_VERSION} -o /tmp/micromamba.tar.bz2 && tar -xjf /tmp/micromamba.tar.bz2 -C /usr/local bin/micromamba
ENV MAMBA_ROOT_PREFIX=/build/mamba
ENV PIP_CACHE_DIR=/build/cache/pip
ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /project
