FROM main-container

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb-xinerama0 \
    libqt5gui5 \
    libxcb-cursor0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy viewer script
COPY viewer.py ./

# Shared memory and X11
VOLUME /dev/shm
VOLUME /tmp/.X11-unix

CMD ["python3", "viewer.py"]
