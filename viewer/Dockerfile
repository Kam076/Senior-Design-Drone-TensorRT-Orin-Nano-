FROM ubuntu:20.04
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Denver

# Dependencies for OpenCV GUI
RUN apt-get update && apt-get install -y \
    python3-pip python3-opencv \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libcanberra-gtk-module libcanberra-gtk3-module \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --no-cache-dir numpy

# Copy viewer script
COPY viewer.py ./

# Shared memory and X11
VOLUME /dev/shm
VOLUME /tmp/.X11-unix

CMD ["python3", "viewer.py"]
