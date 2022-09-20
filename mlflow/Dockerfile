FROM python:3-slim

WORKDIR /mlflow
RUN pip install --no-cache-dir mlflow
EXPOSE 5050

CMD mlflow server --host 0.0.0.0 --port 5050
