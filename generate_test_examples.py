"""Generate test_examples.npz for each disease without re-running the full training notebook."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data_processing.disease_dataset_process import DataProcessor, CATEGORICAL_COLUMNS
from models_classes.mlp_disease_neural_net import ArbovirosesMLP

MODELS_DIR = Path(__file__).resolve().parent / 'models_saved'

for disease in ['dengue', 'chikungunya']:
    disease_dir = MODELS_DIR / disease
    if not (disease_dir / 'artifacts.json').exists():
        print(f'Skipping {disease}: no artifacts.json found')
        continue

    print(f'\n=== {disease.upper()} ===')

    # Load and process data (same pipeline as training)
    dp = DataProcessor(type_disease=disease)
    df, categorical_columns, _ = dp.load_data_process()
    numerical_columns = df.drop(columns=list(categorical_columns) + ['final_classification']).columns

    # Prepare tensors (same as ArbovirosesMLP._prepare_data)
    categorical_tensors, numerical_tensors, target_tensor, _ = ArbovirosesMLP._prepare_data(df, categorical_columns)

    # Split with same random_state=42 to get the same test set
    _, x_test_cat, _, x_test_num, _, y_test = train_test_split(
        categorical_tensors, numerical_tensors, target_tensor,
        test_size=0.1, shuffle=True, random_state=42
    )

    # Sample 200 random rows
    n = min(200, len(y_test))
    idx = np.random.choice(len(y_test), n, replace=False)

    np.savez(
        disease_dir / 'test_examples.npz',
        x_cat=x_test_cat[idx].cpu().numpy(),
        x_num=x_test_num[idx].cpu().numpy(),
        y=y_test[idx].cpu().numpy(),
    )
    print(f'Saved {n} test examples to {disease_dir / "test_examples.npz"}')

print('\nDone!')
