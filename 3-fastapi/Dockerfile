FROM chck/container4ml-jupyter:1.0 AS trainer

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.8-slim

ENV APP_HOME /app

# Install production dependencies.
RUN apt update && apt install -y --no-install-recommends \
    git \
    build-essential \
 && apt clean \
 && rm -rf /var/lib/apt/lists/* \
 && pip install -U pip && pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export --without-hashes -f requirements.txt -o requirements.txt \
 && pip install -r requirements.txt --no-cache-dir

# via multi-stage build: Copy model binary to the container image.
WORKDIR /models
COPY --from=trainer /app/*.bin .

# Copy local code to the container image.
WORKDIR ${APP_HOME}
COPY main.py .

EXPOSE 80

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD ["gunicorn", "main:app", "--bind=0.0.0.0:80", "--workers=1", "--threads=8", "--timeout=0", "--worker-class=uvicorn.workers.UvicornWorker"]
