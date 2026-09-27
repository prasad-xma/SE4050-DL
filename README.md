# SE4050 Deep Learning Assignment

This project currently contains only the work for data preprocessing and the first model, an MLP.

## Folder structure

```text
data/          Original dataset and the processed dataset created by Notebook 01
notebooks/     Preprocessing and MLP training notebooks
models/        Preprocessing objects and the trained MLP model
requirements.txt
```

## Run order

1. Place `CICIDS2017_combined.csv` inside `data/`.
2. Run `notebooks/01_Data_Preprocessing.ipynb` from top to bottom.
3. Run `notebooks/02_MLP_Model.ipynb` from top to bottom.

Notebook 01 creates one processed dataset: `data/processed_data.npz`.

Notebook 02 saves the trained model as `models/mlp_model.pt`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```
