FROM main-container

WORKDIR /app



#Vision Processing
COPY vision_c.py ./vision_c.py
#COPY converter.py ./converter.py
COPY models ./vision_c/models

#Shared Mem
VOLUME /dev/shm

#CMD ["python3","converter.py"]
CMD ["python3","vision_c.py"]
