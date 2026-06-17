#!/usr/bin/env bash
# Render benchmark_report.md -> benchmark_report.pdf.
# Regenerates figures first, then runs pandoc + xelatex.
#
# Note: this node's texlive lacks lmodern.sty, which pandoc's default LaTeX
# template loads unconditionally. We strip that one line into a temp template
# and use Liberation system fonts via the fontspec (xelatex) path.
set -euo pipefail
cd "$(dirname "$0")"

python make_figures.py

TMPL="$(mktemp --suffix=.tex)"
trap 'rm -f "$TMPL"' EXIT
pandoc -D latex | sed '/lmodern/d' > "$TMPL"

pandoc benchmark_report.md -o benchmark_report.pdf \
    --template="$TMPL" --pdf-engine=xelatex \
    -V geometry:margin=1in -V colorlinks=true \
    -V mainfont="Liberation Serif" \
    -V monofont="Liberation Mono" \
    -V sansfont="Liberation Sans"

echo "wrote $(pwd)/benchmark_report.pdf"
