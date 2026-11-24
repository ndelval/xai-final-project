#!/bin/sh
# Compile reports/report.tex with pdfLaTeX. Two passes resolve citations and references.
# Needs a TeX distribution (e.g. TeX Live or MacTeX) with the packages listed in report.tex.
set -eu
cd "$(dirname "$0")"

if ! command -v pdflatex >/dev/null 2>&1; then
    echo "pdfLaTeX is required. Install a TeX distribution and add pdflatex to PATH." >&2
    exit 127
fi

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' 0
trap 'exit 130' INT
trap 'exit 143' HUP TERM

for pass in 1 2; do
    if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory="$BUILD_DIR" report.tex >"$BUILD_DIR/compile-output.log" 2>&1; then
        cat "$BUILD_DIR/compile-output.log" >&2
        exit 1
    fi
done

cp "$BUILD_DIR/report.pdf" report.pdf
echo "reports/report.pdf compiled from reports/report.tex (pdfLaTeX, two passes)"
