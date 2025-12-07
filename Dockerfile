# Dockerfile

# Use the python:3.9-slim base image for a small container size
FROM python:3.9-slim

# Set the application directory inside the container
WORKDIR /app

# Copy and install dependencies first (for faster caching)
COPY requirements.txt /app/

# Install dependencies using the container's Python environment
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (main.py, mnist_model.h5)
COPY . /app/

# The command to run the application using Uvicorn in production mode
# --host 0.0.0.0 is mandatory for the container to be accessible
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]