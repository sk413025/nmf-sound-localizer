.PHONY: all generate validate review-prepare review-gate deploy release clean

REPO_ROOT := $(abspath ..)
PYTHONPATH_PREFIX = PYTHONPATH="$(REPO_ROOT):$${PYTHONPATH}"

all: generate validate deploy review-prepare

generate:
	cd "$(REPO_ROOT)" && $(PYTHONPATH_PREFIX) python -m figures.build.pipeline generate

validate:
	cd "$(REPO_ROOT)" && $(PYTHONPATH_PREFIX) python -m figures.build.pipeline validate

review-prepare:
	cd "$(REPO_ROOT)" && $(PYTHONPATH_PREFIX) python scripts/paper/review_paper_assets.py prepare

review-gate:
	cd "$(REPO_ROOT)" && $(PYTHONPATH_PREFIX) python scripts/paper/review_paper_assets.py gate

deploy: validate
	cd "$(REPO_ROOT)" && $(PYTHONPATH_PREFIX) python -m figures.build.pipeline deploy

release:
	cd "$(REPO_ROOT)" && $(PYTHONPATH_PREFIX) python -m figures.build.pipeline release

clean:
	find output -type f ! -name '.gitignore' -delete
	find review_artifacts -type f ! -name '.gitignore' -delete
