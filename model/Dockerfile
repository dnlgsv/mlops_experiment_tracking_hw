FROM python:3.7-slim

WORKDIR /model
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt


CMD python3 train.py