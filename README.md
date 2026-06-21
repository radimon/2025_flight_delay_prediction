# ConvLSTM Mobility Smart Parking System

An intelligent parking guidance system that integrates spatiotemporal crowd-flow prediction, LLM-based natural language parsing, and multi-criteria route optimization. The system estimates parking availability probability from human mobility data — without relying on physical sensors — and generates optimized navigation routes with fallback mechanisms before departure.

Validated on real-world mobility data from Sapporo City, Hokkaido, Japan (ACM SIGSPATIAL GIS Cup 2025, CityC dataset).

---

## System Overview

The system adopts a dual-layer offline-online architecture:

**Offline Phase**
- Preprocess raw mobility grid data and extract spatiotemporal features
- Train a ConvLSTM-based crowd-flow prediction model
- Simulate parking lot demand and calibrate availability probability via Platt Scaling

**Online Phase**
1. **Spatial Semantic Parsing** — GPT-4 Function Calling extracts origin, destination, date, and time from natural language input
2. **Geographic Retrieval** — Geocoding API converts addresses to coordinates; nearby parking lots retrieved via OpenStreetMap
3. **Dynamic Prediction** — ConvLSTM model estimates parking availability at predicted arrival time
4. **Multi-criteria Scoring** — Candidates ranked by weighted composite score (availability probability, walking distance, detour time, driving time, price)
5. **Route Planning** — A single chained driving route (origin → parking lot 1 → parking lot 2 → ... → final lot) is generated via Google Directions API with waypoints, alongside individual walking routes from each lot to the destination

---

## Structure

```
ConvLSTM_Mobility_Smart_Parking_System/
├── data/
│   ├── raw/                        # Raw CSV files (ignored by git)
│   ├── processed/                  # Processed parquet outputs (ignored by git)
│   ├── parking/                    # Parking lot POI data
│   ├── parking lots/               # Simulated parking lot datasets
│   └── predictions/                # Model prediction outputs
├── models/                         # Trained model checkpoints (.pkl, .joblib)
│   ├── convlstm_sapporo.pkl
│   ├── lstm_sapporo.pkl
│   ├── gru_best.pkl
│   ├── lgbm_best.joblib
│   └── predrnn_best.pkl
├── notebooks/                      # Exploratory analysis & experiments
│   ├── 01_explore_density.ipynb
│   ├── 02_sapporo_18pm_heatmap.ipynb
│   ├── 03_sapporo_spatiotemporal_model.ipynb
│   ├── 03-5_model_test_pipeline_with_ConvLSTM.ipynb
│   ├── 04_query_confidence.ipynb
│   ├── 05_query_centered_coverage_calibration.ipynb
│   ├── 06_LLM_query.ipynb
│   ├── 07_grid_classify_and_parking_data_create.ipynb
│   ├── 07-5_parking_simulate.ipynb
│   ├── 08_generate_parking_params.ipynb
│   ├── 09_google_map.ipynb
│   ├── 10_model_comparison_with_overfitting_trend.ipynb
│   └── 11_pics_for_report.ipynb
├── outputs/                        # Output figures
├── src/                            # Core source modules
│   ├── geo/                        # Grid ↔ lat/lng conversion
│   ├── aggregation.py
│   ├── confidence_engine.py
│   ├── ConvLSTM.py
│   ├── google_maps_engine.py
│   ├── GRU.py
│   ├── LGBM.py
│   ├── LSTM.py
│   ├── parking_engine.py
│   ├── parks_choose.py
│   ├── parks_create.py
│   ├── parks_routing.py
│   ├── PredRNN.py
│   ├── preprocess.py
│   ├── routing_engine.py
│   ├── str_similarity.py
│   └── util.py
├── main.py                         # Pipeline entry point
├── requirements.txt
└── README.md
```

---

## Models

Five spatiotemporal prediction models are compared:

| Model | Description |
|-------|-------------|
| **ConvLSTM** | Primary model — captures spatial neighborhood and temporal dependencies jointly |
| LightGBM | Gradient boosting baseline with engineered lag features |
| LSTM | Sequential baseline (temporal only) |
| GRU | Lightweight recurrent baseline |
| PredRNN | Extended spatiotemporal memory transfer |

ConvLSTM achieves the best generalization (lowest Test RMSE and Test-Train gap) among all models.

---

## Python Version

```bash
python --version
# Recommended: Python 3.12
```

---

## Install Dependencies

Run inside VSCode terminal (CMD, not PowerShell).


```bash  *Run this command on Windows Powershell*
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

```bash
# Remove existing venv if needed
rmdir /s /q venv312

# Create virtual environment
py -3.12 -m venv venv312

# Activate (it will have a (venv312) in front of the directory)
venv312\Scripts\activate.bat

# Upgrade pip
py -m pip install --upgrade pip

# Install PyTorch (skip if already installed)
.\venv312\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Data

```
data/raw/CityC_Sapporo.csv
```

Human mobility grid data from the ACM SIGSPATIAL GIS Cup 2025 (CityC dataset), provided by LY Corporation. Contains anonymized user location records across a 194×201 grid at 30-minute intervals over 75 days (September–November 2019, Sapporo, Japan).

---

## Environment Variables

Create a `.env` file in the project root with the following keys:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
```

## Run

**Command-line mode**
```bash
python main.py
```

**Interactive Web UI (recommended)**
```bash
streamlit run app.py
```

## Developer

**Radimon** — bjhijhijhi573@gmail.com  
**Dymension**