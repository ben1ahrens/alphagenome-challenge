#!/usr/bin/env bash
# ==============================================================
# setup.sh  –  AlphaGenome Serova Pipeline
# Creates the conda environment and installs all prerequisites,
# including cloning and installing the AlphaGenome SDK.
# ==============================================================
set -euo pipefail

ENV_NAME="alphagenome-serova"
PYTHON_VERSION="3.11"

echo "============================================="
echo " AlphaGenome Serova – Environment Setup"
echo "============================================="

# ---- Check for conda ----------------------------------------
if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# ---- Create / update conda env ------------------------------
if conda env list | grep -q "^${ENV_NAME}"; then
    echo "[1/5] Conda environment '${ENV_NAME}' already exists – updating..."
    conda env update --name "${ENV_NAME}" --file environment.yml --prune
else
    echo "[1/5] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda env create --name "${ENV_NAME}" --file environment.yml
fi

# ---- Activate env (source for script context) ---------------
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "[2/5] Installing pip dependencies from requirements.txt..."
pip install -r requirements.txt --quiet

# ---- Install AlphaGenome from GitHub -------------------------
echo "[3/5] Cloning and installing AlphaGenome SDK..."
if [ -d "./alphagenome_sdk" ]; then
    echo "  AlphaGenome SDK directory already exists – pulling latest..."
    git -C ./alphagenome_sdk pull
else
    git clone https://github.com/google-deepmind/alphagenome.git ./alphagenome_sdk
fi
pip install -e ./alphagenome_sdk --quiet
echo "  AlphaGenome SDK installed."

# ---- Create directory structure ------------------------------
echo "[4/5] Creating project directories..."
mkdir -p data/raw data/reference data/processed results/figures

# ---- Copy .env template if not present ----------------------
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "  ⚠️  A .env file has been created. Please add your AlphaGenome API key:"
    echo "     1. Visit https://deepmind.google.com/science/alphagenome"
    echo "     2. Register for a free non-commercial API key"
    echo "     3. Edit .env and set ALPHA_GENOME_API_KEY=<your_key>"
fi

echo "[5/5] Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and add your ALPHA_GENOME_API_KEY"
echo "  2. Place input files in data/raw/:"
echo "       data/raw/VCF_File.gz"
echo "       data/raw/Example_RNA.xlsx"
echo "  3. Download hg38 reference genome:"
echo "       bash scripts/download_reference.sh"
echo "  4. Run the pipeline:"
echo "       conda activate ${ENV_NAME}"
echo "       python src/pipeline.py --config config.yaml"
echo ""
echo "Or explore interactively:"
echo "       jupyter notebook notebooks/exploratory_analysis.ipynb"
