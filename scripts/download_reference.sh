#!/usr/bin/env bash
# ==============================================================
# scripts/download_reference.sh
# Download hg38 reference genome from UCSC Genome Browser.
# Required for pyfaidx-based VCF validation (optional for API mode).
# ==============================================================
set -euo pipefail

OUT_DIR="data/reference"
HG38_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
OUT_FA="${OUT_DIR}/hg38.fa"
OUT_GZ="${OUT_DIR}/hg38.fa.gz"

mkdir -p "${OUT_DIR}"

if [ -f "${OUT_FA}" ]; then
    echo "Reference genome already exists: ${OUT_FA}"
    exit 0
fi

echo "Downloading hg38 FASTA from UCSC (~3 GB)..."
echo "URL: ${HG38_URL}"
wget -c -O "${OUT_GZ}" "${HG38_URL}"

echo "Decompressing..."
gunzip "${OUT_GZ}"

echo "Indexing with samtools faidx..."
samtools faidx "${OUT_FA}"

echo "Done. Reference genome: ${OUT_FA}"
echo "Index file: ${OUT_FA}.fai"
