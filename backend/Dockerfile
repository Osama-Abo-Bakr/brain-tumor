FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (required for PIL / OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend.py ./backend.py

# Copy model to MATCH your MODEL_PATH
COPY backend/models ./backend/models

# Expose API port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]