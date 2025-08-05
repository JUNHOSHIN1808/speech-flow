# --- Base Image ---
FROM python:3.11-slim

# --- Set Working Directory ---
WORKDIR /app

# --- Install System Dependencies for KoNLPy ---
RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get clean

# --- Copy and Install Python Dependencies ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy Application Code and Data ---
COPY . .

# --- Expose Port ---
EXPOSE 8501

# --- Run Application ---
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]