FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENV BACKEND_URL=http://localhost:5000/predict

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
