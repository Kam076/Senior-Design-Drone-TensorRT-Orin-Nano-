FROM ubuntu:20.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver

# System deps
RUN apt-get update && apt-get install -y \
    python3-pip python3-opencv v4l-utils \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --no-cache-dir numpy

# Copy camera script
COPY camera.py ./

# Shared memory
VOLUME /dev/shm

CMD ["python3", "camera.py"]
