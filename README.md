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
# run inside vscode cmd not powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process #在window powershell跑

rmdir /s /q venv312

py -3.12 -m venv venv312

venv312\Scripts\activate.bat #啟用venv

py -m pip install --upgrade pip

.\venv312\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 #已經載過就不用載了

pip install -r requirements.txt

py main.py
```

## Data
data/raw/CityC_Sapporo.csv

## Run
python main.py

## Developer
Radimon, Dymension
