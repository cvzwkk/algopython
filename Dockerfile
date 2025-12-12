FROM python:3.10-bullseye

RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
