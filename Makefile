PYTHON=python
DATA_FILE=IMDB\ Dataset.csv
MODEL_FILE=model/sentiment_model.pkl
VECTORIZER_FILE=model/vectorizer.pkl

.PHONY: help train clean run docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make train         Train the model (requires dataset)"
	@echo "  make run           Run the FastAPI app locally"
	@echo "  make docker-build  Build the Docker image"
	@echo "  make docker-run    Run the Docker container"
	@echo "  make clean         Remove generated files"

train:
	@echo "Training sentiment model..."
	$(PYTHON) train.py

run:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sentiment-api .

docker-run:
	docker run -d -p 8000:8000 sentiment-api --name sentiment-api

clean:
	rm -f $(MODEL_FILE) $(VECTORIZER_FILE)
