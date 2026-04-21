FROM main-container

WORKDIR /app

# Copy camera script
COPY camera.py ./

# Shared memory
VOLUME /dev/shm

CMD ["python3", "camera.py"]
