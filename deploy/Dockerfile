FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt || true

EXPOSE 8000
CMD ["python", "-m", "chonkforge", "serve", "--port", "8000", "--host", "0.0.0.0"]
