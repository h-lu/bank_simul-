FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "central_bank_simulator.py", "--server.port=8501", "--server.address=0.0.0.0"]