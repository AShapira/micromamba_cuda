# CUDA userspace + micromamba. Works with --gpus all at runtime.
FROM mambaorg/micromamba:1.5.10-jammy-cuda-12.2.2

ARG MAMBA_DOCKERFILE_ACTIVATE=0
ARG ENV_NAME=gis

# Use plain bash during build steps
SHELL ["/bin/bash", "-lc"]

# Copy env and create it
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -n ${ENV_NAME} -f /tmp/environment.yml && micromamba clean -a -y

USER root
# Install Git (and SSH client for SSH remotes)
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client ca-certificates && rm -rf /var/lib/apt/lists/*
# Make env default for interactive shells (as root)
RUN echo "micromamba activate ${ENV_NAME}" >> /etc/bash.bashrc
USER $MAMBA_USER

# Switch to entrypoint wrapper for subsequent layers
SHELL ["/usr/local/bin/_entrypoint.sh", "bash", "-lc"]

# Useful runtime env for geospatial + CUDA
ENV PROJ_NETWORK=ON \
    GDAL_CACHEMAX=1024 \
    RMM_POOL_SIZE=2GB \
    UCX_TLS=cuda_copy,cuda_ipc,rc,tcp,sockcm \
    UCX_SOCKADDR_TLS_PRIORITY=sockcm

WORKDIR /workspace
EXPOSE 8888
CMD ["sleep", "infinity"]
