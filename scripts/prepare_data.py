from pathlib import Path

from src.aggregation import build_density_table, save_density


RAW_PATH = Path("data/raw/CityC_Sapporo.csv")
OUTPUT_PATH = Path("data/processed/sapporo_density.parquet")


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_PATH}\n"
            "Download the City C dataset from the SIGSPATIAL GIS Cup 2025 "
            "dataset and place it at data/raw/CityC_Sapporo.csv."
        )

    print(f"Reading raw mobility data from: {RAW_PATH}")
    density = build_density_table(RAW_PATH)

    print(f"Saving processed density data to: {OUTPUT_PATH}")
    save_density(density, OUTPUT_PATH)

    print("Done.")
    print(f"Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()