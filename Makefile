# Define all the targets that are not files/directories
.PHONY: test

test:
	@pytest ./tests/ --cov-report term-missing --cov=graphgen

install:
	@pip install -r requirements.txt
