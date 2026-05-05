# use build.sh and run.sh for building and running 

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH 
# ENV PATH="/usr/local/cuda/bin:${PATH}"
# ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends git build-essential python3-dev libopenblas-dev \
    python3-pip ninja-build -y \
    && rm -rf /var/lib/apt/lists/*

# solve LibX11.so.6 not found error
RUN apt-get update \
    && apt-get install gconf-service libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 \
    libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 \
    libnspr4 libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 \
    libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 ca-certificates \
    fonts-liberation libappindicator1 libnss3 lsb-release xdg-utils wget libxcomposite1 libxcursor1 -y\
    && rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME=omcl_user
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && useradd --no-log-init -u ${USER_ID} -m -g ${USER_NAME} ${USER_NAME}
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
RUN pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

ENV PATH=/home/${USER_NAME}/.local/bin${PATH:+:${PATH}}
ENV PYTHONPATH=/home/${USER_NAME}/.local/lib/python3.10/site-packages:$PYTHONPATH


WORKDIR /home/${USER_NAME}
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin && cd kaolin && git checkout v0.18.0
RUN cd kaolin && pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
RUN cd kaolin && MAX_JOBS=8 FORCE_CUDA=1 python3 setup.py install --user
# just installation check
RUN python3 -c "import kaolin as kal; kal._C.ops.spc.points_to_morton_cuda"

# CUDA12.1 is not supported, perform cpu only installation
# TODO: downgrade cuda version
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"
# for newer pip version:
# RUN pip install --no-build-isolation -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps --config-settings="--install-option=--force_cuda" --config-settings="--install-option=--blas=openblas"
RUN MAX_JOBS=8 pip install --no-build-isolation -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps --install-option="--blas=openblas"

RUN mkdir omcl
COPY docker/requirements.txt omcl/requirements.txt
RUN pip install -r omcl/requirements.txt && rm omcl/requirements.txt 
WORKDIR /home/${USER_NAME}/omcl

ENV HF_HOME=/home/${USER_NAME}/omcl/.cache