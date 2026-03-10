.PHONY: all generate validate deploy clean

all: generate validate deploy

generate:
	python -m figures.build.pipeline generate

validate:
	python -m figures.build.pipeline validate

deploy: validate
	python -m figures.build.pipeline deploy

clean:
	rm -rf output/
