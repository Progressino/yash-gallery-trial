FROM python:3.11-slim

# Parent working dir so 'backend' is a proper Python package
WORKDIR /srv

# System deps for pyarrow / prophet
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libarchive-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source as a package directory
COPY . backend/

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
