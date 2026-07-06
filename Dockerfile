# Use the official lightweight Python 3.11 image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PYTHONPATH=/app/src

# Set the working directory inside the container
WORKDIR /app

# Copy dependency configuration files
COPY pyproject.toml README.md /app/

# Upgrade pip and install package dependencies (including the server extras)
# along with Google Cloud Firestore, GCS, and Firebase Admin Python SDKs
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[server] google-cloud-firestore google-cloud-storage firebase-admin

# Copy codebase
COPY src/ /app/src/
COPY server/ /app/server/

# Ensure directory for local replays exists as a fallback
RUN mkdir -p /app/replays

# Expose port (Cloud Run defaults to 8080)
EXPOSE 8080

# Start the spectator server using Uvicorn
# We use shell form or exec to dynamically bind to the $PORT env variable provided by Cloud Run
CMD exec uvicorn server.main:app --host 0.0.0.0 --port ${PORT}
