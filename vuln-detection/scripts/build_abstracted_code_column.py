"""
Build abstracted_code_v1 next to normalized_code for Case Study 1.

Run in Colab/repo after placing normalization_v3.py in src/case_study_1/.
This script does not alter labels, manifests, or train/test splits.
"""

from pathlib import Path
import pandas as pd

from case_study_1.normalization_v3 import (
    add_abstracted_code_column,
    representation_summary,
    ABSTRACTION_VERSION,
)

# Adjust these to your Drive paths if needed.
PROCESSED_DIR = Path("/content/drive/MyDrive/IntelligentSystemProject/VulnerabilityDetectionData/processed")
INPUT_PARQUET = PROCESSED_DIR / "rdiversevul_cs1_normalized_v1.parquet"
OUTPUT_PARQUET = PROCESSED_DIR / "rdiversevul_cs1_normalized_plus_abstracted_v1.parquet"
SUMMARY_JSON = PROCESSED_DIR / "rdiversevul_cs1_abstracted_v1_summary.json"


def main() -> None:
    if not INPUT_PARQUET.is_file():
        raise FileNotFoundError(INPUT_PARQUET)
    if OUTPUT_PARQUET.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {OUTPUT_PARQUET}")

    df = pd.read_parquet(INPUT_PARQUET)
    if "normalized_code" not in df.columns:
        raise KeyError("Expected normalized_code column")

    df_out = add_abstracted_code_column(
        df,
        source_column="normalized_code",
        target_column="abstracted_code_v1",
    )
    df_out.to_parquet(OUTPUT_PARQUET, index=False)

    summary = {
        "abstraction_version": ABSTRACTION_VERSION,
        "input_path": str(INPUT_PARQUET),
        "output_path": str(OUTPUT_PARQUET),
        "normalized_code": representation_summary(df_out["normalized_code"]),
        "abstracted_code_v1": representation_summary(df_out["abstracted_code_v1"]),
    }
    pd.Series(summary).to_json(SUMMARY_JSON, indent=2)
    print("Saved:", OUTPUT_PARQUET)
    print("Saved:", SUMMARY_JSON)
    print(summary)


if __name__ == "__main__":
    main()
