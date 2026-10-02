FROM python:3.14-slim
WORKDIR /eidos
COPY . /eidos
RUN pip install --no-cache-dir .
RUN eidos -h
