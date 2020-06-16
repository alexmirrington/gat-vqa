# Define all the targets that are not files/directories
.PHONY: test

test:
	@pytest --cov=graphgen
