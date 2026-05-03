FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY models/ models/
COPY configs/ configs/
COPY README.md .
COPY model_card.md .

EXPOSE 7860
CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "7860"]