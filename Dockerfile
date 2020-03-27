FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip


RUN pip3 install numpy pandas matplotlib seaborn sklearn


WORKDIR /app

COPY student_performance.py .
COPY student/* ./student/

CMD ["python3","-u","./student_performance.py"]