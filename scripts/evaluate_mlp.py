from pathlib import Path
import json
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn


PROJECT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT / 'data' / 'processed_data.npz'
MODELS_DIR = PROJECT / 'models'
RESULTS_DIR = PROJECT / 'results' / 'mlp'
MODEL_FILE = MODELS_DIR / 'mlp_model.pt'
LABEL_ENCODER_FILE = MODELS_DIR / 'label_encoder.joblib'
BATCH_SIZE = 1024


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, features):
        return self.layers(features)


def main():
    for required_file in (DATA_FILE, MODEL_FILE, LABEL_ENCODER_FILE):
        if not required_file.exists():
            raise FileNotFoundError(f'Missing required file: {required_file}')

    label_encoder = joblib.load(LABEL_ENCODER_FILE)
    class_names = list(label_encoder.classes_)
    number_of_classes = len(class_names)
    all_labels = np.arange(number_of_classes)
    benign_index = class_names.index('BENIGN')

    processed_data = np.load(DATA_FILE, allow_pickle=True)
    X_test = processed_data['X_test'].astype(np.float32)
    y_test = processed_data['y_test'].astype(np.int64)
    if X_test.ndim != 2 or y_test.ndim != 1 or len(X_test) != len(y_test):
        raise ValueError('X_test and y_test have incompatible shapes.')
    if X_test.shape[1] == 0 or np.any((y_test < 0) | (y_test >= number_of_classes)):
        raise ValueError('The shared test split is incompatible with the label encoder.')

    model = MLP(X_test.shape[1], number_of_classes)
    checkpoint = torch.load(MODEL_FILE, map_location='cpu')
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    predictions = []
    probability_batches = []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(X_test), BATCH_SIZE):
            batch = torch.from_numpy(X_test[start:start + BATCH_SIZE])
            probabilities = torch.softmax(model(batch), dim=1)
            probability_batches.append(probabilities.numpy())
            predictions.append(probabilities.argmax(dim=1).numpy())
    inference_seconds = time.perf_counter() - inference_start
    test_predictions = np.concatenate(predictions)
    test_probabilities = np.concatenate(probability_batches)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_predictions, labels=all_labels, zero_division=0,
    )
    per_class_metrics = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
    })

    multiclass_metrics = {
        'accuracy': accuracy_score(y_test, test_predictions),
        'macro_precision': precision_recall_fscore_support(
            y_test, test_predictions, average='macro', zero_division=0,
        )[0],
        'macro_recall': precision_recall_fscore_support(
            y_test, test_predictions, average='macro', zero_division=0,
        )[1],
        'macro_f1': f1_score(y_test, test_predictions, average='macro', zero_division=0),
        'weighted_precision': precision_recall_fscore_support(
            y_test, test_predictions, average='weighted', zero_division=0,
        )[0],
        'weighted_recall': precision_recall_fscore_support(
            y_test, test_predictions, average='weighted', zero_division=0,
        )[1],
        'weighted_f1': f1_score(y_test, test_predictions, average='weighted', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, test_predictions),
    }

    y_test_binary = (y_test != benign_index).astype(int)
    predictions_binary = (test_predictions != benign_index).astype(int)
    attack_score = 1.0 - test_probabilities[:, benign_index]
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_test_binary, predictions_binary, labels=[0, 1],
    ).ravel()
    binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
        y_test_binary, predictions_binary, average='binary', zero_division=0,
    )
    binary_metrics = {
        'precision': binary_precision,
        'recall_detection_rate': binary_recall,
        'f1': binary_f1,
        'roc_auc': roc_auc_score(y_test_binary, attack_score),
        'pr_auc': average_precision_score(y_test_binary, attack_score),
        'false_positive_rate': false_positive / max(false_positive + true_negative, 1),
        'mcc': matthews_corrcoef(y_test_binary, predictions_binary),
        'true_negative': int(true_negative),
        'false_positive': int(false_positive),
        'false_negative': int(false_negative),
        'true_positive': int(true_positive),
    }

    confusion = confusion_matrix(y_test, test_predictions, labels=all_labels)
    confusion_table = pd.DataFrame(confusion, index=class_names, columns=class_names)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_class_metrics.to_csv(RESULTS_DIR / 'per_class_metrics.csv', index=False)
    confusion_table.to_csv(RESULTS_DIR / 'confusion_matrix.csv')
    pd.DataFrame({'y_true': y_test, 'y_pred': test_predictions}).to_csv(
        RESULTS_DIR / 'test_predictions.csv', index=False,
    )
    np.savez_compressed(
        RESULTS_DIR / 'test_predictions.npz',
        y_true=y_test,
        y_pred=test_predictions,
        probabilities=test_probabilities.astype(np.float32),
        class_names=np.array(class_names),
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_table,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.title('MLP confusion matrix - shared test split')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()

    metrics = {
        'model': 'MLP / DNN',
        'evaluation_split': 'shared X_test and y_test from data/processed_data.npz',
        'class_names': class_names,
        'multiclass': {key: float(value) for key, value in multiclass_metrics.items()},
        'binary_benign_vs_attack': {
            key: (float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in binary_metrics.items()
        },
        'efficiency': {
            'training_seconds': None,
            'inference_seconds': inference_seconds,
            'inference_ms_per_1000_samples': 1000 * 1000 * inference_seconds / len(X_test),
            'trainable_parameters': int(parameter_count),
            'model_file_bytes': MODEL_FILE.stat().st_size,
            'model_file_mb': MODEL_FILE.stat().st_size / 1024 ** 2,
            'batch_size': BATCH_SIZE,
            'device': 'cpu',
        },
    }
    with open(RESULTS_DIR / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)

    print('MLP evaluation complete')
    print(f"Accuracy: {multiclass_metrics['accuracy']:.6f}")
    print(f"Macro-F1: {multiclass_metrics['macro_f1']:.6f}")
    print(f"Binary F1: {binary_metrics['f1']:.6f}")
    print(f'Inference seconds: {inference_seconds:.6f}')
    print(f'Trainable parameters: {parameter_count:,}')
    print(f'Results saved to: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
