FROM python:3.10 AS BUILD

RUN apt-get update -y

RUN pip install --upgrade pip

COPY . /opt/app

COPY requirements.txt /opt/app

WORKDIR /opt/app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "src/main.py"]