"""
tests/test_parse_vcf.py – Unit tests for VCF parsing utilities.

Run with:  pytest tests/test_parse_vcf.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import parse_csq_header, parse_csq_entry, strip_version


# ------------------------------------------------------------------ #
# strip_version                                                        #
# ------------------------------------------------------------------ #

class TestStripVersion:
    def test_strips_version(self):
        assert strip_version("ENSG00000157764.13") == "ENSG00000157764"

    def test_no_version(self):
        assert strip_version("ENSG00000157764") == "ENSG00000157764"

    def test_empty(self):
        assert strip_version("") == ""


# ------------------------------------------------------------------ #
# parse_csq_header                                                     #
# ------------------------------------------------------------------ #

EXAMPLE_HEADER_LINES = [
    "##fileformat=VCFv4.2",
    '##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP. '
    'Format: Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|'
    'HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|Existing_variation|'
    'DISTANCE|STRAND|FLAGS|VARIANT_CLASS|SYMBOL_SOURCE|HGNC_ID|CANONICAL|MANE_SELECT|MANE_PLUS_CLINICAL|'
    'TSL|APPRIS|CCDS|ENSP|SWISSPROT|TREMBL|UNIPARC|UNIPROT_ISOFORM|RefSeq|GENE_PHENO|SIFT|PolyPhen|'
    'DOMAINS|miRNA|HGVS_OFFSET|AF|AFR_AF|AMR_AF|EAS_AF|EUR_AF|SAS_AF|gnomADe_AF|gnomADe_AFR_AF|'
    'gnomADe_AMR_AF|gnomADe_ASJ_AF|gnomADe_EAS_AF|gnomADe_FIN_AF|gnomADe_NFE_AF|gnomADe_OTH_AF|'
    'gnomADe_SAS_AF|gnomADg_AF|gnomADg_AFR_AF|gnomADg_AMI_AF|gnomADg_AMR_AF|gnomADg_ASJ_AF|'
    'gnomADg_EAS_AF|gnomADg_FIN_AF|gnomADg_MID_AF|gnomADg_NFE_AF|gnomADg_OTH_AF|gnomADg_SAS_AF|'
    'MAX_AF|MAX_AF_POPS|FREQS|CLIN_SIG|SOMATIC|PHENO|PUBMED|MOTIF_NAME|MOTIF_POS|HIGH_INF_POS|'
    'MOTIF_SCORE_CHANGE|TRANSCRIPTION_FACTORS|LoF|LoF_filter|LoF_flags|LoF_info|SpliceAI_pred_DP_AG|'
    'SpliceAI_pred_DP_AL|SpliceAI_pred_DP_DG|SpliceAI_pred_DP_DL|SpliceAI_pred_DS_AG|'
    'SpliceAI_pred_DS_AL|SpliceAI_pred_DS_DG|SpliceAI_pred_DS_DL|SpliceAI_pred_SYMBOL">',
]


class TestParseCsqHeader:
    def test_extracts_fields(self):
        fields = parse_csq_header(EXAMPLE_HEADER_LINES)
        assert "Consequence" in fields
        assert "Gene" in fields
        assert "SYMBOL" in fields
        assert "CANONICAL" in fields

    def test_empty_lines(self):
        fields = parse_csq_header(["##fileformat=VCFv4.2"])
        assert fields == []


# ------------------------------------------------------------------ #
# parse_csq_entry                                                      #
# ------------------------------------------------------------------ #

class TestParseCsqEntry:
    FIELDS = ["Allele", "Consequence", "IMPACT", "SYMBOL", "Gene", "Feature_type",
               "Feature", "BIOTYPE", "EXON", "INTRON", "HGVSc", "HGVSp",
               "cDNA_position", "CDS_position", "Protein_position", "Amino_acids",
               "Codons", "Existing_variation", "DISTANCE", "STRAND", "FLAGS",
               "VARIANT_CLASS", "SYMBOL_SOURCE", "HGNC_ID", "CANONICAL"]

    def test_single_transcript(self):
        csq = "A|missense_variant|MODERATE|TP53|ENSG00000141510|Transcript|ENST00000269305|protein_coding|" + \
              "|".join([""] * (len(self.FIELDS) - 8))
        records = parse_csq_entry(csq, self.FIELDS)
        assert len(records) == 1
        assert records[0]["Gene"] == "ENSG00000141510"
        assert records[0]["Consequence"] == "missense_variant"

    def test_multiple_transcripts(self):
        t1 = "A|missense_variant|MODERATE|TP53|ENSG00000141510|Transcript|ENST00000269305|protein_coding|" + \
             "|".join([""] * (len(self.FIELDS) - 8))
        t2 = "A|downstream_gene_variant|MODIFIER|TP53|ENSG00000141510|Transcript|ENST00000413465|protein_coding|" + \
             "|".join([""] * (len(self.FIELDS) - 8))
        csq = f"{t1},{t2}"
        records = parse_csq_entry(csq, self.FIELDS)
        assert len(records) == 2

    def test_padding_on_short_entry(self):
        # Fewer pipe-separated values than expected fields
        csq = "A|downstream_gene_variant|MODIFIER|TNFRSF18"
        records = parse_csq_entry(csq, self.FIELDS)
        assert len(records) == 1
        # Missing fields should be empty string
        assert records[0].get("CANONICAL", "") == ""


# ------------------------------------------------------------------ #
# Integration: parse_vcf on real file                                  #
# ------------------------------------------------------------------ #

class TestParseVcfIntegration:
    """
    Integration test – runs the full parse_vcf step on the real data file.
    Skipped unless the data file is present (useful in CI without large files).
    """

    @pytest.mark.skipif(
        not Path("data/raw/VCF_File.gz").exists(),
        reason="data/raw/VCF_File.gz not present",
    )
    def test_runs_without_error(self, tmp_path):
        import importlib.util, shutil
        spec = importlib.util.spec_from_file_location(
            "parse_vcf", Path("src/1_parse_vcf.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cfg = {
            "data": {"vcf_gz": "data/raw/VCF_File.gz"},
            "vcf": {
                "vcf_inner_path": (
                    "ff1df4a9-2318-4dba-8f34-cb69dde4360c/"
                    "TCGA_LUAD.f368273c-bd2d-4b97-97ec-a04cb130af1e."
                    "wgs.GATK4_MuTect2_Pair.somatic_annotation.vcf.gz"
                ),
                "filter_pass_only": True,
                "max_indel_length": 50,
                "vep_csq_field": "CSQ",
                "genome_build": "GRCh38",
            },
            "output": {"results_dir": str(tmp_path)},
        }
        df = mod.run(cfg)
        assert len(df) > 0
        assert "variant_id" in df.columns
        assert "gene_id" in df.columns
