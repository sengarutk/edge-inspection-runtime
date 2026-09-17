#!/bin/bash
set -e
cd /home/sengar/edge-quality-intelligence/docs/paper

echo "=== Attempting compilation with texlive/texlive:latest ==="
if docker run --rm -v "$(pwd):/work" -w /work texlive/texlive:latest /bin/bash -c "
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  bibtex main && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
"; then
  echo "=== Compilation successful with texlive/texlive:latest ==="
  exit 0
fi

echo "=== Attempting fallback compilation with blang/latex:ubuntu ==="
if docker run --rm -v "$(pwd):/work" -w /work blang/latex:ubuntu /bin/bash -c "
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  bibtex main && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex && \
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
"; then
  echo "=== Compilation successful with blang/latex:ubuntu ==="
  exit 0
fi

echo "=== Docker compilation failed or images not reachable ==="
exit 1
