.PHONY: docs

init:
	pip install -r requirements.txt

test:
	python -m unittest discover

docs:
	cd docs && make html
	@echo "\nBuild successful!"
	@echo "View the docs homepage at docs/_build/html/index.html.\n"
