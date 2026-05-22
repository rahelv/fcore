FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .
COPY static/ static/

# Mount point for the image dataset (read-only)
VOLUME ["/home/ubuntu/data/robust_dataset"]

EXPOSE 5000
CMD ["python3", "app.py"]