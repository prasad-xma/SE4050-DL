# SE4050 Deep Learning Assignment

## Project

**Network intrusion detection using the CICIDS2017 dataset**

The current project scope is:

1. Prepare one shared processed dataset.
2. Train the first model, a Multilayer Perceptron (MLP).
3. Save the trained model for later comparison with the other group models.

## Folder structure

```text
SE4050-DL/
├── .venv/                       Local Python environment
├── data/
│   ├── CICIDS2017_combined.csv  Original dataset
│   └── processed_data.npz       Created by Notebook 01
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   └── 02_MLP_Model.ipynb
├── models/
│   ├── preprocessor.joblib      Created by Notebook 01
│   ├── label_encoder.joblib     Created by Notebook 01
│   └── mlp_model.pt             Created by Notebook 02
├── requirements.txt
└── README.md
```

Only the original CSV exists before preprocessing. The other files shown above are created when the notebooks run.

## Environment setup

The `.venv` is already configured on the main project computer. To activate it:

```powershell
cd C:\Users\prasa\Documents\SLIIT\DL\SE4050-DL
.\.venv\Scripts\Activate.ps1
```

Each group member should create their own `.venv` instead of copying this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The `.venv` folder and large dataset files are excluded from GitHub.

## Start Jupyter

After activating the environment:

```powershell
jupyter notebook
```

The installed PyTorch version currently uses the CPU. Training will work, but it will be slower than training with a CUDA GPU.

## Notebook 01: data preprocessing

Run `notebooks/01_Data_Preprocessing.ipynb` first. Its tasks are separated into clearly numbered steps:

1. Load the original CSV.
2. Clean the column names.
3. Remove duplicated columns.
4. Remove duplicated rows.
5. Group the original attack labels into nine classes.
6. Separate the features and target.
7. Convert features to numeric values.
8. Replace infinity values with missing values.
9. Split the data into 80% training and 20% testing sets.
10. Fill missing values using training-set medians.
11. Remove constant features.
12. Standardize features using training-set statistics.
13. Encode the class labels.
14. Save the processed dataset and preprocessing objects.

The notebook creates one processed dataset:

```text
data/processed_data.npz
```

It contains:

- `X_train`
- `y_train`
- `X_test`
- `y_test`

The original CSV is never modified.

## Notebook 02: MLP model

Run `notebooks/02_MLP_Model.ipynb` after preprocessing. It:

1. Loads `processed_data.npz`.
2. Converts the arrays into PyTorch tensors.
3. Creates training and test data loaders.
4. Builds the MLP architecture.
5. Calculates class weights for the imbalanced dataset.
6. Trains the model for 15 epochs.
7. Displays the training-loss graph.
8. Calculates accuracy, macro-F1 and per-class metrics.
9. Displays the confusion matrix.
10. Saves the trained model.

The MLP architecture is:

```text
Input features
    ↓
256 neurons + ReLU + Dropout
    ↓
128 neurons + ReLU + Dropout
    ↓
64 neurons + ReLU
    ↓
9 output classes
```

The trained model is saved as:

```text
models/mlp_model.pt
```

## Group model workflow

All four group models should use the same `processed_data.npz` file and the same train/test split:

```text
processed_data.npz
├── MLP
├── Denoising Autoencoder
├── TabNet
└── FT-Transformer
```

Each member trains and saves their model separately in `models/`. The models are not automatically combined. After training, the group compares them using the same test set and metrics such as macro-F1, recall, accuracy and confusion matrices. The best model can then be selected as the final intrusion-detection model.

## Important notes

- Run Notebook 01 before Notebook 02.
- Do not fit preprocessing separately for each model.
- Do not change the shared test set.
- Accuracy alone is not sufficient because CICIDS2017 is highly imbalanced.
- Ensure the computer has enough free disk space and memory before processing the complete dataset.
