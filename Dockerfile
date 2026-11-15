FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

ENV FASTAPI_HOST=0.0.0.0
CMD ["fastapi", "run", "app/main.py"]