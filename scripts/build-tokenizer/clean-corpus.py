from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from cleaning import clean_dataframe, normalize_text, prepare_document

REPO = "bakrianoo/jabarti-llm-dataset"

OUTPUT_DIR=Path(__file__).parent/"outputs"
MAX_CHUNKS=20

TRAIN_SPLITS = {"phase1_train", "phase2_train"}
EVAL_SPLITS = {"phase1_eval", "phase2_eval"}




HF_FILES = {
    "phase1_train": "pretrain/phase1_train-00000-of-00001.parquet",
    "phase1_eval":  "pretrain/phase1_eval-00000-of-00001.parquet",

    "phase2_train": "pretrain/phase2_train-00000-of-00001.parquet",
    "phase2_eval":  "pretrain/phase2_eval-00000-of-00001.parquet",

    "ft_train":     "finetune/train-00000-of-00001.parquet",
    "ft_eval":      "finetune/eval-00000-of-00001.parquet",
}



def download_and_filter():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

    for name,file_name in HF_FILES.items():

        print(f"Downloading {name} from HuggingFace Hub...")
        Local = hf_hub_download(repo_id=REPO, repo_type="dataset",filename=file_name)

        df=pd.read_parquet(Local)

        if name in TRAIN_SPLITS:
            len_before=len(df)
            # Filter out rows with empty text or label
            df=(
                df.sort_values(["article_id", "chunk_index"])
                .groupby('article_id',sort=False)
                .head(MAX_CHUNKS)
                .reset_index(drop=True)
                )
            len_after=len(df)
            print(f"Filtered {len_before-len_after} rows from {name} due to exceeding {MAX_CHUNKS} chunks per article.")

        file_name=f"{name}_filtered.parquet"

        df.to_parquet(OUTPUT_DIR/file_name, index=False)
        print(f"Saved filtered {name} to {OUTPUT_DIR/file_name}")

def clean_phase(phase, input_path, output_path):

    print(f"==== clean_phase: {phase} ====")

    df = pd.read_parquet(input_path)

    print(f"  Source        : {input_path}")
    print(f"  Rows          : {len(df):,}")

    cleaned_df = clean_dataframe(df)

    removed_rows = len(df) - len(cleaned_df)
    print(f"  Rows kept        : {len(cleaned_df):,}  (dropped {removed_rows:,} empty rows)")

    cleaned_df.to_parquet(output_path, index=False)



def main():
    download_and_filter()
    print("All files downloaded and filtered successfully.")

    for name in [*TRAIN_SPLITS, *EVAL_SPLITS]:
        input_path = OUTPUT_DIR / f"{name}_filtered.parquet"
        output_path = OUTPUT_DIR / f"{name}_clean.parquet"

        clean_phase(phase=name, input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()





        