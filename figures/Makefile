.PHONY: all generate validate review-prepare review-gate deploy release clean

all: generate validate deploy review-prepare

generate:
	python -m figures.build.pipeline generate

validate:
	python -m figures.build.pipeline validate

review-prepare:
	python ../scripts/paper/review_paper_assets.py prepare

review-gate:
	python ../scripts/paper/review_paper_assets.py gate

deploy: validate
	python -m figures.build.pipeline deploy

release:
	python -m figures.build.pipeline release

clean:
	rm -rf output/ review_artifacts/
