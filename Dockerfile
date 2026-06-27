FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any are needed in the future
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and configure the working directory
RUN useradd -u 1000 -m appuser && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy requirements and install packages
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all source files
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 8000

# Run uvicorn server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
