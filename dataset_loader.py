"""
import pandas as pd

COMMON_PROMPT_COLS = ["prompt", "text", "input", "attack", "message"]
COMMON_LABEL_COLS = ["label", "type", "is_attack"]

def load_and_prepare_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[loader] dataset shape: {df.shape}")

    # Auto-detect text and label columns
    text_col = next((c for c in COMMON_PROMPT_COLS if c in df.columns), None)
    label_col = next((c for c in COMMON_LABEL_COLS if c in df.columns), None)

    if not text_col:
        raise RuntimeError("Could not detect prompt column")

    # Canonicalize labels
    def canonicalize_label(val):
        if pd.isna(val):
            return "unknown"
        s = str(val).lower()
        if s in ["1","true","attack","yes","y"]:
            return "attack"
        if s in ["0","false","benign","no","n"]:
            return "benign"
        return "unknown"

    df["prompt"] = df[text_col]
    df["label"] = df[label_col].apply(canonicalize_label) if label_col else "unknown"

    return df[["prompt","label"]]
"""

# dataset_loader.py
import pandas as pd

def load_and_prepare_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[loader] dataset shape: {df.shape}")

    # Use "user" as the prompt column
    if "user" not in df.columns:
        raise RuntimeError("Expected 'user' column in dataset")

    # Create labels: if attack_family is NaN/empty → benign, else → attack
    def canonicalize_label(val):
        if pd.isna(val) or str(val).strip() == "":
            return "benign"
        return "attack"

    df["prompt"] = df["user"]
    df["label"] = df["attack_family"].apply(canonicalize_label) if "attack_family" in df.columns else "unknown"

    return df[["id", "prompt", "label", "attack_family", "secret"]]
