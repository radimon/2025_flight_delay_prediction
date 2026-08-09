from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from scripts.train_convlstm import prepare_training_data
from src.ConvLSTM import load_convlstm_embed, predict_convlstm_embed


DATA_PATH = Path("data/processed/sapporo_density.parquet")
MODEL_PATH = Path("models/convlstm_sapporo.pkl")


def main():
    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {DATA_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Rebuild the same train / val / test data
    # --------------------------------------------------------

    print("Loading processed data...")

    df = pd.read_parquet(DATA_PATH)

    print(f"Loaded {len(df):,} rows.")
    print("Rebuilding test set...")

    train_df, val_df, test_df, priB = prepare_training_data(df)

    print(f"Test rows: {len(test_df):,}")

    # --------------------------------------------------------
    # Load trained ConvLSTM
    # --------------------------------------------------------

    print("Loading trained ConvLSTM...")

    model, cfg = load_convlstm_embed(MODEL_PATH)

    print("Model loaded.")

    # --------------------------------------------------------
    # Same lookup used during the original experiment
    # --------------------------------------------------------

    lookup_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Predict test set
    # --------------------------------------------------------

    print("Running test-set prediction...")

    y_pred = predict_convlstm_embed(
        model,
        test_df,
        seq_len=cfg.get("seq_len", 8),
        patch_radius=cfg.get("patch_radius", 4),
        batch_size=512,
        use_residual=True,
        lookup_df=lookup_df,
        freq_prior_df=priB,
    )

    y_true = test_df["count"].to_numpy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    print()
    print("=" * 50)
    print("ConvLSTM Test Results")
    print("=" * 50)
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print("=" * 50)

    print()
    print("Reference results from presentation:")
    print("RMSE : 1.922")
    print("MAE  : 1.215")
    print("R²   : 0.716")


if __name__ == "__main__":
    main()