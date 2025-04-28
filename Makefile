# 0. Setting default
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

message ?= Default-commit-message
level ?= patch

# 1. Default help command to list available Sphinx options
help:
	@echo "Available commands:"
	@echo "  help       - Show this help message"
	@echo "  html       - Generate HTML documentation with sphinx-build (output at docs/build/html/)"
	@echo "  latexpdf   - Generate LaTeX PDF documentation with Sphinx sphinx-build (output at docs/build/latex/)"
	@echo "  clean      - Clean the documentation build directory docs/build/"
	@echo "  bump       - Update the version of the package (default: patch, use level=major/minor/patch)"
	@echo "  git        - Commit and push changes to master (use message='Your commit message')"
	@echo "  app        - Build the application with PyInstaller (output at dist/)"
	@echo "  test       - Run the tests of the package with pytest"

.PHONY: help Makefile

# 2. Generate HTML documentation
html:
	$(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

# 3. Generate LaTeX PDF documentation
latexpdf:
	$(SPHINXBUILD) -b latex $(SOURCEDIR) $(BUILDDIR)/latex
	cd $(BUILDDIR)/latex && pdflatex pydistort.tex && pdflatex pydistort.tex

# 4. Clean the documentation
clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O);
	cd $(BUILDDIR); mkdir -p html; mkdir -p latex

# 5. Update the version of the package
bump:
	bumpver update --$(level) --no-fetch

# 6. Git Push origin Master
git:
	git checkout master
	git add -A .
	git commit -m "$(message)"
	git push origin master

# 7. Create the application
app:
	echo "from pydistort.__main__ import __main_gui__" > run_gui.py
	echo "__main_gui__()" >> run_gui.py
	pyinstaller --name pydistort --onefile --windowed run_gui.py
	rm run_gui.py
	rm -rf build
	rm run_gui.spec

# 8. Tests the package
test:
	pytest tests
