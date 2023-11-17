FROM python:3.10 AS BUILD

RUN apt-get update -y

RUN pip install --upgrade pip

COPY . /opt/app

COPY requirements.txt /opt/app

WORKDIR /opt/app

RUN pip install -r requirements.txt

WORKDIR /opt/app/src

RUN python setup.py build_ext --inplace

ENTRYPOINT ["python", "main.py"]