FROM python:3.15-slim

WORKDIR /my_ml_project

COPY requirments.txt .

RUN pip install --no-cache-dir -r requirments.txt

COPY . .

CMD ["python", "predict.py"]
