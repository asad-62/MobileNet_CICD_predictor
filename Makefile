.PHONY: check test syntax api ui health metrics eval clean \
	docker-build minikube-load helm-install helm-upgrade helm-uninstall \
	k8s-status k8s-logs k8s-restart ingress-test prometheus-test \
	prometheus-forward grafana-forward

IMAGE_NAME=face-type-recommender
IMAGE_TAG=prometheus-v1
NAMESPACE=face-type-ai
RELEASE=face-type-ai
CHART=./helm/face-type-ai

check: syntax test

test:
	PYTHONPATH=. pytest -q

syntax:
	python -m py_compile app/main.py app/inference.py app/gradio_app.py app/monitoring.py app/schemas.py app/recommender.py app/config.py

api:
	PYTHONPATH=. uvicorn app.main:api --reload --host 0.0.0.0 --port 7860

ui:
	PYTHONPATH=. python -m app.gradio_app

health:
	curl -s http://localhost:7860/health | python -m json.tool

metrics:
	curl -s http://localhost:7860/metrics | python -m json.tool

eval:
	PYTHONPATH=. python scripts/evaluate_model.py

docker-build:
	docker build -f Dockerfile.local -t $(IMAGE_NAME):$(IMAGE_TAG) .

minikube-load:
	minikube image load $(IMAGE_NAME):$(IMAGE_TAG)

helm-install:
	helm install $(RELEASE) $(CHART) -n $(NAMESPACE) --create-namespace --set image.tag=$(IMAGE_TAG)

helm-upgrade:
	helm upgrade $(RELEASE) $(CHART) -n $(NAMESPACE) --set image.tag=$(IMAGE_TAG)

helm-uninstall:
	helm uninstall $(RELEASE) -n $(NAMESPACE)

k8s-status:
	kubectl get pods -n $(NAMESPACE)
	kubectl get svc -n $(NAMESPACE)
	kubectl get ingress -n $(NAMESPACE)
	helm list -n $(NAMESPACE)

k8s-logs:
	kubectl logs -n $(NAMESPACE) deploy/$(IMAGE_NAME) --tail=100

k8s-restart:
	kubectl rollout restart deployment $(IMAGE_NAME) -n $(NAMESPACE)
	kubectl rollout status deployment $(IMAGE_NAME) -n $(NAMESPACE)

ingress-test:
	curl -s http://face-type-ai.local/health | python -m json.tool

prometheus-test:
	curl -s http://face-type-ai.local/prometheus | grep "face_"

prometheus-forward:
	kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090

grafana-forward:
	kubectl port-forward -n monitoring svc/monitoring-grafana 3001:80

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete