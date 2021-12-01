# Use the official lightweight Python image.
# https://hub.docker.com/_/python
#FROM python:3.8
FROM python:3.8-slim
#FROM python:3.8-alpine

ENV APP_HOME /app

RUN apt update && apt install -y --no-install-recommends \
    curl \
 && apt clean \
 && rm -rf /var/lib/apt/lists/* \
 && pip install -U pip && pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes -f requirements.txt -o requirements.txt \
 && pip install -r requirements.txt --no-cache-dir

WORKDIR ${APP_HOME}
COPY main.py .
#COPY . .

CMD ["python", "main.py"]
