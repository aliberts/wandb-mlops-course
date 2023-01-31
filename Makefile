setup-poetry:
	curl -sSL https://install.python-poetry.org | python3 -
dataset:
	mkdir artifacts && cd artifacts && \
		curl https://storage.googleapis.com/wandb_course/bdd_simple_1k.zip -o bdd_simple_1k.zip && \
		while [ "`find . -type f -name '*.zip' | wc -l`" -gt 0 ]; \
		do find -type f -name "*.zip" -exec unzip -- '{}' \; \
		-exec rm -- '{}' \;; done
	mv artifacts/BDD_SIMPLE_1k artifacts/bdd_simple_1k:v0
