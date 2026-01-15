# Makefile for Interspeech 2026 paper
# Usage: make pdf / make docx / make clean

PANDOC = pandoc
DRAFT = draft.md
METADATA = metadata.yaml
BIB = references.bib
OUTPUT_DIR = output

# Default target
all: pdf

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Markdown to PDF
pdf: $(OUTPUT_DIR)
	$(PANDOC) $(DRAFT) -o $(OUTPUT_DIR)/draft.pdf \
		--metadata-file=$(METADATA) \
		--pdf-engine=xelatex \
		--highlight-style=tango \
		-V colorlinks=true \
		-V linkcolor=blue \
		-V urlcolor=blue

# Markdown to PDF with citations (when references.bib exists)
pdf-cite: $(OUTPUT_DIR)
	$(PANDOC) $(DRAFT) -o $(OUTPUT_DIR)/draft.pdf \
		--metadata-file=$(METADATA) \
		--citeproc --bibliography=$(BIB) \
		--pdf-engine=xelatex \
		--highlight-style=tango

# Markdown to Word
docx: $(OUTPUT_DIR)
	$(PANDOC) $(DRAFT) -o $(OUTPUT_DIR)/draft.docx \
		--metadata-file=$(METADATA)

# Markdown to Word with citations
docx-cite: $(OUTPUT_DIR)
	$(PANDOC) $(DRAFT) -o $(OUTPUT_DIR)/draft.docx \
		--metadata-file=$(METADATA) \
		--citeproc --bibliography=$(BIB)

# Markdown to HTML (for preview)
html: $(OUTPUT_DIR)
	$(PANDOC) $(DRAFT) -o $(OUTPUT_DIR)/draft.html \
		--metadata-file=$(METADATA) \
		--standalone \
		--mathjax

# Word count
count:
	@echo "Word count (excluding code blocks):"
	@cat $(DRAFT) | grep -v '```' | wc -w

# Clean output
clean:
	rm -rf $(OUTPUT_DIR)

# Watch for changes (requires fswatch on macOS)
watch:
	fswatch -o $(DRAFT) $(METADATA) | xargs -n1 -I{} make pdf

.PHONY: all pdf pdf-cite docx docx-cite html count clean watch
