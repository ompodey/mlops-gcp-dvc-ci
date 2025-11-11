# ---------- Dockerfile ----------

# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code (app.py, model, etc.)
COPY . .

# Expose Flask port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]
