# Base image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Work inside app project
WORKDIR /app/HW2_GitProxy

# System deps for faster wheels (httptools/uvloop)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer cache)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project source
COPY . /app/HW2_GitProxy

# Default runtime env (override at run time)
ENV PORT=5000
EXPOSE 5000

# Simple healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=2).getcode()==200 else 1)"]

# Run FastAPI app (module path relative to WORKDIR)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
