import json
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

import sys
PROJECT = Path.cwd()
sys.path.append(str(PROJECT))

from src.models.ft_transformer import FTTransformer

def main():
    data_file = PROJECT / 'data' / 'processed_data.npz'
    model_file = PROJECT / 'models' / 'ft_transformer_model.pt'
    label_encoder_file = PROJECT / 'models' / 'label_encoder.joblib'
    results_dir = PROJECT / 'results' / 'ft_transformer'
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    X_test = data['X_test']
    y_test = data['y_test']

    label_encoder = joblib.load(label_encoder_file)
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)
    n_features = X_test.shape[1]

    print(f"Features: {n_features}, Classes: {n_classes} ({class_names})")
    print(f"Test samples: {len(y_test):,}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    # Build model
    model = FTTransformer(
        n_features=n_features,
        n_classes=n_classes,
        d_token=64,
        n_blocks=3,
        n_heads=4,
        ffn_factor=2.0,
        attention_dropout=0.15,
        ffn_dropout=0.15,
        residual_dropout=0.10,
        head_dropout=0.20,
        head_hidden_dim=64
    )

    print(f"Loading state dict from {model_file}...")
    state_dict = torch.load(model_file, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_file_size_mb = model_file.stat().st_size / (1024 * 1024)
    print(f"Model Parameters: {trainable_params:,} | Model Size: {model_file_size_mb:.4f} MB")

    # DataLoader
    batch_size = 4096
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Running inference on test set...")
    all_preds = []
    all_probs = []
    all_targets = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(batch_y.numpy())

    inference_seconds = time.perf_counter() - start_time
    ms_per_1000 = (inference_seconds / len(y_test)) * 1000 * 1000

    y_pred = np.concatenate(all_preds)
    y_probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_targets)

    print(f"Inference completed in {inference_seconds:.2f}s ({ms_per_1000:.2f} ms/1000 flows)")

    # 1. Multiclass Metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_p = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_r = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("\n--- Multiclass Metrics ---")
    print(f"Accuracy:          {acc:.4%}")
    print(f"Balanced Accuracy: {bal_acc:.4%}")
    print(f"Macro Precision:   {macro_p:.4%}")
    print(f"Macro Recall:      {macro_r:.4%}")
    print(f"Macro F1:          {macro_f1:.4%}")
    print(f"Weighted F1:       {weighted_f1:.4%}")

    # 2. Per-class metrics
    p_per_class, r_per_class, f1_per_class, s_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )

    per_class_df = pd.DataFrame({
        'precision': p_per_class,
        'recall': r_per_class,
        'f1': f1_per_class,
        'support': s_per_class
    }, index=class_names)
    per_class_csv_path = results_dir / 'per_class_metrics.csv'
    per_class_df.to_csv(per_class_csv_path)
    print(f"\nPer-class metrics saved to {per_class_csv_path}")
    print(per_class_df)

    # 3. Binary Benign vs Attack Metrics
    # In label encoder, find index of BENIGN
    benign_idx = class_names.index('BENIGN')
    y_true_binary = (y_true != benign_idx).astype(int) # 0 = BENIGN, 1 = ATTACK
    y_pred_binary = (y_pred != benign_idx).astype(int)
    attack_probs = 1.0 - y_probs[:, benign_idx]

    bin_cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = bin_cm.ravel()

    bin_precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    bin_recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    bin_f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    bin_roc_auc = roc_auc_score(y_true_binary, attack_probs)
    bin_pr_auc = average_precision_score(y_true_binary, attack_probs)
    bin_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    bin_mcc = matthews_corrcoef(y_true_binary, y_pred_binary)

    print("\n--- Binary Benign vs Attack Metrics ---")
    print(f"Precision:         {bin_precision:.4%}")
    print(f"Recall (Det Rate): {bin_recall:.4%}")
    print(f"F1 Score:          {bin_f1:.4%}")
    print(f"ROC-AUC:           {bin_roc_auc:.4%}")
    print(f"PR-AUC:            {bin_pr_auc:.4%}")
    print(f"FPR:               {bin_fpr:.4%}")
    print(f"MCC:               {bin_mcc:.4f}")
    print(f"TN: {tn:,}, FP: {fp:,}, FN: {fn:,}, TP: {tp:,}")

    # 4. Save test predictions and probabilities
    test_preds_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    test_preds_path = results_dir / 'test_predictions.csv'
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"\nTest predictions saved to {test_preds_path}")

    probs_path = results_dir / 'test_probabilities.npy'
    np.save(probs_path, y_probs)
    print(f"Test probabilities saved to {probs_path}")

    # 5. Confusion Matrix CSV & Plot
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(results_dir / 'confusion_matrix.csv')

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('FT-Transformer Multiclass Confusion Matrix (Test Split)', fontsize=14, pad=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {results_dir / 'confusion_matrix.png'}")

    # 6. Save Config
    config_dict = {
        "model": "FT-Transformer",
        "n_features": n_features,
        "n_classes": n_classes,
        "d_token": 64,
        "n_blocks": 3,
        "n_heads": 4,
        "ffn_factor": 2.0,
        "attention_dropout": 0.15,
        "ffn_dropout": 0.15,
        "residual_dropout": 0.10,
        "head_dropout": 0.20,
        "head_hidden_dim": 64,
        "batch_size": 1024,
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "epochs": 15,
        "optimizer": "AdamW",
        "loss": "Weighted CrossEntropyLoss"
    }
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # 7. Comprehensive metrics.json
    metrics_json = {
        "model": "FT-Transformer",
        "config": config_dict,
        "multiclass": {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1)
        },
        "binary_benign_vs_attack": {
            "precision": float(bin_precision),
            "recall_detection_rate": float(bin_recall),
            "f1": float(bin_f1),
            "roc_auc": float(bin_roc_auc),
            "pr_auc": float(bin_pr_auc),
            "false_positive_rate": float(bin_fpr),
            "mcc": float(bin_mcc),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp)
        },
        "efficiency": {
            "trainable_parameters": int(trainable_params),
            "model_file_mb": float(model_file_size_mb),
            "training_seconds": 3120.0, # ~52 mins recorded during Colab training
            "inference_seconds": float(inference_seconds),
            "inference_ms_per_1000_rows": float(ms_per_1000),
            "epochs_run": 15,
            "device": str(device)
        }
    }

    metrics_json_path = results_dir / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics JSON saved to {metrics_json_path}")
    print("\nFT-Transformer evaluation complete and artifacts exported successfully!")

if __name__ == '__main__':
    main()
