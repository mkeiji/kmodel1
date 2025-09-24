PYTHON=python
NAIVE_BAYES_PATH=models/naive_bayes
HUGGING_FACE_PATH=models/hugging_face

.PHONY: help train clean run docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make train-1       Train the naive_bayes (requires dataset)"
	@echo "  make train-2       Train the hugging_face (requires dataset)"
	@echo "  make evaluate      Evaluate the trained models (requires trained models)"
	@echo "  make run           Run the FastAPI app locally"
	@echo "  make docker-build  Build the Docker image"
	@echo "  make docker-run    Run the Docker container"
	@echo "  make clean         Remove generated files"

train-1:
	@echo "Training naive_bayes model..."
	$(PYTHON) trainings/train.py

train-2:
	@echo "Training hugging_face model..."
	$(PYTHON) trainings/train_transformer.py

evaluate:
	@echo "Evaluating models..."
	$(PYTHON) evaluation/evaluate_models.py

run:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker build -t kmodel1 .

docker-run:
	docker run --rm -d -p 8000:8000 kmodel1 --name kmodel1-app

clean:
	rm -rf $(NAIVE_BAYES_PATH)
	rm -rf $(HUGGING_FACE_PATH)
