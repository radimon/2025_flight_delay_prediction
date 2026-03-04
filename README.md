# 2025_japan_people_flow

## Structure
2025_japan_people_flow/
├── data/
│   ├── raw/            # raw CSV files (ignored by git)
│   └── processed/      # processed parquet outputs (ignored by git)
├── notebooks/          # exploratory analysis & visualization
├── src/
│   └── aggregation.py  # core data processing logic
├── main.py             # pipeline entry point
├── requirements.txt
└── README.md

## Python version
```bash
python --version
```

## Install Dependencies
```bash
python -m pip install -r requirements.txt
```
for venv
```bash
.\venv312\Scripts\python.exe -m pip install --upgrade pip
.\venv312\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Data
data/raw/CityC_Sapporo.csv

## Run
python main.py

## Developer
Radimon, Dymension