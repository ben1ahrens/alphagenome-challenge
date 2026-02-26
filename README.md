# AlphaGenome Variant-Expression Pipeline

**Serova Tech Challenge · Benjamin Ahrens**

This pipeline uses [AlphaGenome](https://github.com/google-deepmind/alphagenome) to predict the regulatory impact of cancer-associated somatic variants on gene expression, and compares these predictions against observed RNA-seq data from the same TCGA-LUAD patient sample.

---

## Biological Motivation

One barrier to personalised cancer vaccine design is determining which neoantigens are actually expressed. RNA sequencing is the gold standard but is costly and slow. If a genomic model can predict which variants are likely to reduce expression of nearby genes, we can prioritise RNA-seq or deprioritise poorly-expressed targets without additional sequencing.

AlphaGenome takes 1 Mb of DNA sequence and predicts thousands of functional genomic tracks—including RNA-seq coverage—at single-base resolution. By comparing predictions under the reference allele versus the tumour alternate allele, we obtain a predicted expression change (∆RNA-seq) for each somatic variant. This pipeline tests how well those predictions correlate with measured TPM values in the same patient.

---

## Data

| File | Description |
|------|-------------|
| `VCF_File.gz` | TCGA-LUAD somatic variants (GRCh38, MuTect2), patient TCGA-05-4384, ~2,774 PASS calls |
| `Example_RNA.xlsx` | TCGA RNA-seq expression (GENCODE v36): TPM, FPKM per gene |

The VCF file is VEP-annotated (CSQ field) and contains Ensembl gene IDs, enabling direct linkage to the RNA expression file.

---

## Pipeline Overview

```
VCF (somatic variants)          RNA expression (TPM)
        │                               │
   1_parse_vcf.py              2_load_rna.py
        │                               │
        └──────────┬────────────────────┘
                   ▼
          3_run_alphagenome.py
          (score each PASS variant;
           RNA-seq + CAGE scorers,
           lung UBERON:0002048)
                   │
                   ▼
          4_aggregate_scores.py
          (gene-level: min ∆RNA-seq
           across variants per gene)
                   │
                   ▼
          5_compare_expression.py
          (merge predicted vs observed;
           stratify TPM >1 vs ≤1;
           Spearman ρ, AUC, plots)
                   │
                   ▼
          6_gtex_context.py
          (fetch GTEx lung medians;
           contextualise against
           normal lung expression)
                   │
                   ▼
          results/summary_report.html
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your_handle>/alphagenome-serova.git
cd alphagenome-serova
bash setup.sh          # creates micromamba env and installs all dependencies
micromamba activate alphagenome-env
```

### 2. Configure API key

Obtain a free non-commercial API key from https://deepmind.google.com/science/alphagenome, then:

```bash
export ALPHA_GENOME_API_KEY="your_key_here"
# Or add to .env (see .env.example)
```

### 3. Download the reference genome (hg38)

AlphaGenome does not require local sequence for API use; however, the pipeline uses pyfaidx for VCF validation. Download hg38 from UCSC:

```bash
bash scripts/download_reference.sh   # ~3 GB, downloads to data/reference/
```

### 4. Place input files

```bash
cp /path/to/VCF_File.gz              data/raw/
cp /path/to/Example_RNA.xlsx         data/raw/
```

### 5. Run the full pipeline

```bash
python src/pipeline.py --config config.yaml
```

Or step-by-step:

```bash
python src/1_parse_vcf.py
python src/2_load_rna.py
python src/3_run_alphagenome.py   # requires API key
python src/4_aggregate_scores.py
python src/5_compare_expression.py
python src/6_gtex_context.py
```

Results are written to `results/`.

---

## Key Assumptions & Caveats

- **Model generalisation**: AlphaGenome is trained on germline variation in healthy tissues. Somatic cancer variants may involve mechanisms (copy-number changes, transcription factor rewiring) not captured by the model.
- **Variant-to-gene assignment**: Each variant is linked to the gene(s) annotated in the VEP CSQ field. For genes with multiple variants, we report the most negative predicted ∆RNA-seq (worst-case approach). An alternative is to sum log-fold changes.
- **Sequence context**: We use a 1 Mb window centred on each variant. Variants near chromosome ends are padded with Ns.
- **Tissue specificity**: We score with lung-relevant tracks (`UBERON:0002048`). AlphaGenome is not fine-tuned for cancer; predictions reflect a regulatory reference landscape.
- **GTEx comparison**: GTEx lung medians represent normal tissue. Tumour expression is expected to diverge.
- **TPM stratification**: TPM > 1 is used as a threshold for "expressed" genes, following standard practice.

---

## Outputs

| File | Description |
|------|-------------|
| `results/vcf_parsed.parquet` | Parsed PASS variants with gene annotations |
| `results/rna_expression.parquet` | Cleaned RNA expression table |
| `results/alphagenome_scores.parquet` | Raw per-variant AlphaGenome scores |
| `results/gene_scores.parquet` | Gene-level aggregated scores |
| `results/merged.parquet` | Merged predicted + observed expression |
| `results/figures/` | Scatter plots, ROC curves, heatmaps |
| `results/summary_report.html` | Auto-generated HTML report |

---

## Citation

```bibtex
@article{alphagenome,
  title={AlphaGenome: advancing regulatory variant effect prediction with a unified DNA sequence model},
  author={Avsec, Žiga and Latysheva, Natasha and Cheng, Jun and Novati, Guido et al.},
  year={2025},
  doi={https://doi.org/10.1101/2025.06.25.661532}
}
```

---

## Project Structure

```
alphagenome-serova/
├── README.md
├── setup.sh                    # micromamba env + pip install
├── requirements.txt            # Python dependencies
├── environment.yml             # micromamba environment file
├── config.yaml                 # pipeline configuration
├── .env.example                # API key template
├── data/
│   ├── raw/                    # input files (gitignored)
│   └── reference/              # hg38 FASTA (gitignored)
├── src/
│   ├── pipeline.py             # orchestrates all steps
│   ├── 1_parse_vcf.py          # VCF parsing & filtering
│   ├── 2_load_rna.py           # RNA-seq expression loading
│   ├── 3_run_alphagenome.py    # AlphaGenome variant scoring
│   ├── 4_aggregate_scores.py   # gene-level aggregation
│   ├── 5_compare_expression.py # comparison & evaluation
│   ├── 6_gtex_context.py       # GTEx contextualisation
│   └── utils.py                # shared utilities
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   └── download_reference.sh
├── tests/
│   ├── test_parse_vcf.py
│   ├── test_load_rna.py
│   └── test_aggregation.py
└── results/                    # auto-generated (gitignored)
```
