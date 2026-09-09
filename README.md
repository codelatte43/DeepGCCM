# DeepGCCM

Python implementation and evaluation suite for spatial causal discovery with GCCM and DeepGCCM.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`rasterio`, `GDAL`, and related geospatial dependencies may require a platform-specific installation.

## Run

```bash
python preprocess_usa.py --diagnose
python run_realworld_experiment.py --quick
python baseline/run_baselines.py --quick
```

## Layout

- `src/`: GCCM, DeepGCCM, data, metrics, plotting, and USA preprocessing code.
- `baseline/`: baseline causal-discovery implementations.

🎉 Accepted by CSAI 2026!