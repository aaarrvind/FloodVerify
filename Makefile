.PHONY: lint test demo
lint:
	ruff check src tests
test:
	pytest -q
demo:
	python demo/app.py
