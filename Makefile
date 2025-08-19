install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py

test:
	pytest --maxfail=1 --disable-warnings -q || true

hf-login:
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

deploy: hf-login
	huggingface-cli upload asad2662/face-type-classifier . \
		--repo-type=space \
		--commit-message "Auto-deploy from GitHub Actions"
