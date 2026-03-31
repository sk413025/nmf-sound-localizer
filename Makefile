.PHONY: paper-build paper-pdf paper-check manuscript figures paper-review-assets paper-review-gate clean

paper-build:
	bash scripts/paper/build_docx.sh

paper-pdf:
	bash scripts/paper/build_pdf.sh

paper-check:
	python scripts/paper/check_required_sections.py
	python scripts/paper/check_figure_references.py
	python scripts/paper/check_governance_links.py
	python scripts/paper/check_asset_boundaries.py
	python scripts/paper/verify_provenance.py
	bash scripts/paper/build_docx.sh

manuscript: figures paper-build

figures:
	$(MAKE) -C figures all

paper-review-assets:
	python scripts/paper/review_paper_assets.py prepare

paper-review-gate:
	python scripts/paper/review_paper_assets.py gate

clean:
	$(MAKE) -C figures clean
