.PHONY: manuscript figures clean

manuscript: figures
	bash scripts/paper/build_docx.sh

figures:
	$(MAKE) -C figures all

clean:
	$(MAKE) -C figures clean
