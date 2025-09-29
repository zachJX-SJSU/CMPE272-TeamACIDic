# ---- Base image ----
FROM python:3.10-slim

# Prevent Python from writing .pyc files & buffering logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# System deps (faster wheels for uvloop, httptools, etc.)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better build cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source
COPY . /app

# Defaults (can be overridden at runtime)
ENV PORT=5000
# Do NOT bake the secret into the image; pass it with -e or --env-file
# ENV WEBHOOK_SECRET=proxy

# Expose the service port
EXPOSE 5000

# Healthcheck (optional, nice for graders)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=2).getcode()==200 else 1)"]


# Run FastAPI app (consolidated path)
CMD ["uvicorn", "HW2_GitProxy.app.main:app", "--host", "0.0.0.0", "--port", "5000"]
