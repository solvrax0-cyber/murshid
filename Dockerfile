FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# pre-download the embedding model + build the index at image build time (fast cold start)
RUN python -m scripts.build_index
ENV PORT=8000
CMD uvicorn app.server:app --host 0.0.0.0 --port ${PORT}
