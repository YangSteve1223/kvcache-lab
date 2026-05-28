#!/bin/bash
# Compile the preprint PDF
# Usage: bash compile.sh

set -e

pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

echo "Compilation complete. Output: main.pdf"
