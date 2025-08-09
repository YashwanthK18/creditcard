# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# copy files
COPY . /app

# install system deps and python libs
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "80"]
