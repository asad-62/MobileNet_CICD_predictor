.PHONY: test api ui health metrics clean syntax check eval
check: syntax test

test:
	PYTHONPATH=. pytest -q

api:
	PYTHONPATH=. uvicorn app.main:api --reload --host 0.0.0.0 --port 7860

ui:
	PYTHONPATH=. python -m app.gradio_app

health:
	curl -s http://localhost:7860/health | python -m json.tool

metrics:
	curl -s http://localhost:7860/metrics | python -m json.tool

syntax:
	python -m py_compile app/main.py app/inference.py app/gradio_app.py app/monitoring.py app/schemas.py app/recommender.py app/config.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


eval:
	PYTHONPATH=. python scripts/evaluate_model.py	