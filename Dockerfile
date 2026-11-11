FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

# only the code the API needs; models/ is mounted at runtime
COPY backend ./backend
COPY ml ./ml

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]