FROM python:3.12-slim

LABEL maintainer="priya"
LABEL description="CAT — meditation script scrapers and scorer"

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    requests>=2.31 \
    beautifulsoup4>=4.12

# Copy source
COPY src/ src/

# Data output directory
RUN mkdir -p /app/data/raw

# Default: show available scrapers
CMD ["python", "-c", "import os; print('Available scrapers:'); [print(' ', f) for f in os.listdir('src') if f.startswith('scrape')]"]
