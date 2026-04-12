FROM python:3.12-slim

LABEL maintainer="priya"
LABEL description="CAT — Cognitive Appraisal Theory study: meditation transcript scraping and scoring"

WORKDIR /app

# Install yt-dlp system dependency (ffmpeg for audio extraction if needed)
RUN apt-get update -q && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY scrape_meditations.py .
COPY scoring/ scoring/
COPY scraping/ scraping/

# Install the package with dependencies
RUN pip install --no-cache-dir ".[dev]"

# Data directories
RUN mkdir -p /app/data/raw /app/data/scored /app/output

# Default: run the scoring pipeline help
CMD ["python", "-m", "scoring.pipeline", "--help"]
