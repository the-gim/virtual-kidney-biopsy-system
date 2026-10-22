#!/usr/bin/env bash
# Script to convert Markdown thesis files to Microsoft Word (.docx)

cd "$(dirname "$0")"

echo "=========================================================="
echo " Converting Virtual Kidney Biopsy Dissertation to Word... "
echo "=========================================================="

# 1. Convert Entire Combined Dissertation
echo "[1/2] Generating Full Combined Dissertation (.docx)..."
pandoc \
  00_Front_Matter.md \
  Chapter_1_Introduction.md \
  Chapter_2_Literature_Review.md \
  Chapter_3_Software_Tools_Evaluation.md \
  Chapter_4_Requirements_Specification.md \
  Chapter_5_Research_Methodology.md \
  Chapter_6_Solution_Design.md \
  Chapter_7_Implementation_Details.md \
  Chapter_8_Results_and_Critical_Evaluation.md \
  Chapter_9_Professional_Legal_Ethical_Issues.md \
  Chapter_10_Conclusions_and_Future_Work.md \
  References.md \
  Appendices.md \
  -o Virtual_Kidney_Biopsy_Full_Dissertation.docx

echo " -> Successfully created: thesis/Virtual_Kidney_Biopsy_Full_Dissertation.docx"

# 2. Convert Individual Chapters
echo "[2/2] Generating individual chapter Word documents..."
for f in Chapter_*.md; do
  output_docx="${f%.md}.docx"
  pandoc "$f" -o "$output_docx"
  echo " -> Converted $f -> $output_docx"
done

echo "=========================================================="
echo " Conversion Complete! All Word files are in the thesis/ folder. "
echo "=========================================================="
