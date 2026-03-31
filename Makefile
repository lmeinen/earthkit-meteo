setup:
	pre-commit install

default: qa tests type-check

qa:
	pre-commit run --all-files

tests:
	python -m pytest -vv --cov=. --cov-report=html
