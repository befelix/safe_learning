doc:
	cd docs && $(MAKE) html

coverage:
	coverage html

test-local:
	./scripts/test_code.sh

test: docker
	docker run safe_learning_py2 make test-local
	docker run safe_learning_py3 make test-local

docker:
	docker build -f Dockerfile.python2 -t safe_learning_py2 .
	docker build -f Dockerfile.python3 -t safe_learning_py3 .

