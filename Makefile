.PHONY: manuscript figures paper-review-assets paper-review-gate clean

manuscript: figures
	bash scripts/paper/build_docx.sh

figures:
	$(MAKE) -C figures all

paper-review-assets:
	python scripts/paper/review_paper_assets.py prepare

paper-review-gate:
	python scripts/paper/review_paper_assets.py gate

clean:
	$(MAKE) -C figures clean
