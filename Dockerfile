# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install necessary system dependencies (useful for building C-extensions in statsmodels/pandas)
RUN apt-get update && apt-get install -y gcc g++ tzdata && rm -rf /var/lib/apt/lists/*

# Force the container to use US Eastern Time so crons align perfectly with market hours
ENV TZ="America/New_York"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install Python dependencies
RUN pip install --no-cache-dir yfinance pandas numpy statsmodels alpaca-trade-api prefect

# Copy the execution flow into the container
COPY src/stat_arb_flow.py /app/stat_arb_flow.py

# The command to run the Prefect flow when the container starts
CMD ["python", "stat_arb_flow.py"]