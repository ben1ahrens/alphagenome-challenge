#!/usr/bin/env bash
# ==============================================================
# setup.sh  –  AlphaGenome Serova Pipeline
# Creates the conda environment and installs all prerequisites,
# including cloning and installing the AlphaGenome SDK.
# ==============================================================
set -euo pipefail

ENV_NAME="alphagenome-env"
PYTHON_VERSION="3.12"

echo "============================================="
echo " AlphaGenome Serova – Environment Setup"
echo "============================================="

# ---- Check for micromamba ----------------------------------------
if ! command -v micromamba &>/dev/null; then
    echo "ERROR: micromamba not found. Please install micromamba first."
    
    exit 1
fi

# ---- Create / update conda env ------------------------------
if micromamba env list | grep -q "${ENV_NAME}"; then
    echo "[1/5] Environment '${ENV_NAME}' already exists – updating..."
    micromamba update --name "${ENV_NAME}" --file environment.yml -y
else
    echo "[1/5] Creating environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    micromamba env create --name "${ENV_NAME}" --file environment.yml -y
fi

# ---- Activate env (source for script context) ---------------
# shellcheck disable=SC1091
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

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
