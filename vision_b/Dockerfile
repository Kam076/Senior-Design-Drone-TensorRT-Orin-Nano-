FROM main-container

WORKDIR /app



#Vision Processing
COPY vision_b.py ./vision_b.py
#COPY converter.py ./converter.py
COPY models ./vision_b/models

#Shared Mem
VOLUME /dev/shm


CMD ["python3","vision_b.py"]
