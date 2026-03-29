FROM nvcr.io/nvidia/l4t-base:r36.2.0

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver

# System deps + Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libc-bin \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + fix numpy FIRST
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir numpy>=1.24.2

# Install ultralytics
RUN pip3 install --no-cache-dir ultralytics==8.0.92

# Copy files
COPY vision_d.py ./vision_d.py
COPY models ./models

CMD ["python3", "vision_d.py"]
