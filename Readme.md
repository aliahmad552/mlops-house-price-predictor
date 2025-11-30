# MLOps House Price Predictor

A complete MLOps pipeline to predict house prices in Pakistan using machine learning models (XGBoost, Random Forest, Decision Tree) with **DVC** for data & model versioning and **MLflow** for experiment tracking.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Requirements](#requirements)
3. [Project Structure](#project-structure)
4. [Setup](#setup)
5. [Step-by-Step Pipeline](#step-by-step-pipeline)
6. [DVC & MLflow Integration](#dvc--mlflow-integration)
7. [Remote Setup (DagsHub)](#remote-setup-dagshub)
8. [Usage](#usage)
9. [Author](#author)

---

## Project Overview

This project implements a full **MLOps workflow**:

- **Data ingestion** from MySQL or CSV
- **Data preprocessing** and transformation with NumPy & scikit-learn
- **Training multiple models** (XGBoost, Random Forest, Decision Tree)
- **Evaluation** with RMSE & R²
- **MLflow** experiment tracking
- **DVC** for versioning datasets, models, and preprocessor objects
- **Remote storage** via DagsHub for artifacts and MLflow experiments

---

## Requirements

- Python >= 3.10
- Libraries:
  - pandas, numpy, scikit-learn
  - xgboost
  - mlflow
  - dvc
  - python-dotenv
  - pyyaml
- MySQL (optional if fetching data from database)

Install requirements:

```bash
pip install -r requirements.txt
```

## Project Structure
mlops-house-price-predictor/
│
├─ data/
│  ├─ raw/                  # Raw CSV data
│  ├─ ingested/             # Ingested train/test splits
│  └─ transformed/          # Preprocessed NumPy arrays
│
├─ artifacts/
│  ├─ preprocessor/
│  └─ models/               # Saved models (sale/rent)
│
├─ src/
│  ├─ config/
│  │  └─ config.yaml
│  ├─ data/
│  │  ├─ data_ingestion.py
│  │  └─ data_preprocessing.py
│  ├─ models/
│  │  ├─ train_sale_model.py
│  │  └─ train_rent_model.py
│  └─ utils/
│     ├─ logger.py
│     ├─ file_ops.py
│     └─ exceptions.py
│
├─ params.yaml
├─ dvc.yaml
├─ .env
└─ README.md

Setup

## Clone the repository:
```bash
git clone https://github.com/aliahmad552/mlops-house-price-predictor.git
cd mlops-house-price-predictor

```
## Create & activate virtual environment:
```bash
python -m venv myenv
myenv\Scripts\activate   # Windows
source myenv/bin/activate # Linux/Mac

```
### Install dependencies:
```bash
pip install -r requirements.txt
```

Set MLflow credentials in .env:
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/aliahmad552/mlops-house-price-predictor.mlflow
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_pat>
```
Step-by-Step Pipeline
### 1️⃣ Data Ingestion

Run the data ingestion script:

python src/data/data_ingestion.py


Fetches raw data from MySQL or CSV

Saves ingested train/test CSVs

## Tracked with DVC
```bash
dvc add data/ingested/train.csv data/ingested/test.csv
git add data/.gitignore data/ingested/*.dvc
git commit -m "Ingested data"
```
## 2️⃣ Data Transformation

Run preprocessing:
```bash
dvc repro
```

Converts data to NumPy arrays (X_train.npy, X_test.npy, y_train.npy, y_test.npy)

Saves preprocessor object (preprocessor.pkl)

## DVC tracks transformed files:
```bash
dvc add data/transformed/X_train.npy data/transformed/X_test.npy data/transformed/y_train.npy data/transformed/y_test.npy artifacts/preprocessor/preprocessor.pkl
git add .
git commit -m "Transformed data and preprocessor"
```
## 3️⃣ Train Sale & Rent Models

Train sale model:
```bash
dvc repro
```

## Train rent model:
```bash
dvc repro
```

Multiple models (XGBoost, Random Forest, Decision Tree) trained

## MLflow tracks:

- Parameters (from params.yaml)

- Metrics (RMSE, R²)

- Models (artifact paths)

### DVC tracks best model:
```bash
dvc add artifacts/models/sale_model.pkl artifacts/models/rent_model.pkl
git add .
git commit -m "Trained sale & rent models"
```
## 4️⃣ Run Full Pipeline
```bash
dvc repro
```

### Automatically runs all stages:

- Data ingestion

- Data transformation

- Train sale model

- Train rent model

## DVC & MLflow Integration

- DVC: version datasets, preprocessor, models

- MLflow: track experiments and metrics

- Remote storage: DagsHub

### Set DVC remote:
```bash
dvc remote add -d dagshub-storage https://dagshub.com/aliahmad552/mlops-house-price-predictor.dvc
dvc push
```

MLflow remote:
```bash
mlflow ui          # local UI
```
 j  # Or push to DagsHub with credentials from .env