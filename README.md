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
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
Remove-Item -Recurse -Force .\venv312
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
py -m pip install --upgrade pip
.\venv312\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data
data/raw/CityC_Sapporo.csv

## Run
python main.py

## Developer
Radimon, Dymension