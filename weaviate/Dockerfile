FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /schema

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY schema.py .

CMD ["python", "schema.py"]
