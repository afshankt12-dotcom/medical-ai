from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent.parent

# Main folders
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"

# ==========================
# Dataset Files
# ==========================

MEDQUAD_PATH = DATASET_DIR / "medquad.csv"

MEDQUAD_HF_PATH = DATASET_DIR / "medquad_hf.parquet"

PUBMEDQA_PATH = DATASET_DIR / "pubmedqa.parquet"

CHATBOT_DATASET_PATH = DATASET_DIR / "medical_chatbot_dataset.csv"

ORIGINAL_PQAL_PATH = DATASET_DIR / "ori_pqal.json"

# Future datasets

PUBMED_LARGE_DIR = DATASET_DIR / "pubmed"

PUBMED_BASELINE_DIR = PUBMED_LARGE_DIR / "baseline"