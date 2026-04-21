FROM main-container

WORKDIR /app

# Copy files

COPY vision_d.py ./vision_d.py
COPY models ./models

#Shared Mem
VOLUME /dev/shm

CMD ["python3", "vision_d.py"]
