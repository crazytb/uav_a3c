#!/bin/bash

# LaTeX Paper Compilation Script for IEEE Transaction Format
# Usage: ./compile.sh [clean]

MAIN_FILE="paper.tex"
BIB_FILE="references.bib"
OUTPUT_PDF="paper.pdf"

echo "🚀 Starting IEEE Transaction Paper Compilation..."

# Check if main tex file exists
if [ ! -f "$MAIN_FILE" ]; then
    echo "❌ Error: $MAIN_FILE not found!"
    exit 1
fi

# Clean option
if [ "$1" == "clean" ]; then
    echo "🧹 Cleaning auxiliary files..."
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls *.synctex.gz
    echo "✅ Clean complete!"
    exit 0
fi

# Create figures directory if it doesn't exist
mkdir -p figures

echo "📝 First LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_FILE"

if [ $? -ne 0 ]; then
    echo "❌ First LaTeX compilation failed!"
    echo "📋 Check the log file for errors."
    exit 1
fi

echo "📚 Processing bibliography..."
if [ -f "$BIB_FILE" ]; then
    bibtex paper
    if [ $? -ne 0 ]; then
        echo "⚠️ BibTeX compilation had issues, continuing..."
    fi
else
    echo "⚠️ No bibliography file found, skipping bibtex..."
fi

echo "📝 Second LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_FILE"

echo "📝 Final LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "📄 Output: $OUTPUT_PDF"
    
    # Show file size
    if [ -f "$OUTPUT_PDF" ]; then
        SIZE=$(ls -lh "$OUTPUT_PDF" | awk '{print $5}')
        echo "📊 File size: $SIZE"
    fi
    
    # Optional: Open PDF (macOS)
    if command -v open &> /dev/null; then
        echo "📖 Opening PDF..."
        open "$OUTPUT_PDF"
    fi
else
    echo "❌ Final LaTeX compilation failed!"
    echo "📋 Check the log file for errors."
    exit 1
fi

echo "🎉 Paper compilation complete!"