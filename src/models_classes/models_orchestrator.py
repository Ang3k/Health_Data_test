import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from data_processing.disease_dataset_process import DataProcessor
from models_classes.lgbm_classifier import GradientBoostingDiseaseClassifier
from models_classes.mlp_disease_neural_net import ArbovirosesMLP, device


class ModelsOrchestrator:

    def __init__(self, type_disease):
        self.data_processor = DataProcessor(type_disease=type_disease)
        self.df, self.categorical_columns, _ = self.data_processor.load_data_process()
        self.numerical_columns = self.df.drop(columns=list(self.categorical_columns) + ['final_classification']).columns
        self.train_loader = None
        self.test_loader = None
        self.y_train = None
        self.y_test = None

    def prepare_data(self):
        categorical_tensors, numerical_tensors, target_tensor, embedding_sizes = ArbovirosesMLP._prepare_data(self.df, self.categorical_columns)
        x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test = train_test_split(
            categorical_tensors, numerical_tensors, target_tensor,
            test_size=0.1, shuffle=True, random_state=42
        )

        self.y_train = y_train
        self.y_test = y_test
        self.train_loader = DataLoader(TensorDataset(x_train_cat, x_train_num, y_train), batch_size=8192, shuffle=True)
        self.test_loader  = DataLoader(TensorDataset(x_test_cat,  x_test_num,  y_test),  batch_size=8192, shuffle=False)

        return x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test, embedding_sizes

    def train_mlp(self, embedding_sizes, save_path, epochs=150, patience=6):
        model = ArbovirosesMLP(
            numericals_shape=len(self.numerical_columns),
            embedding_sizes=embedding_sizes,
            hidden_layers=[1024, 512, 256, 128],
            probability_dropout=[0.1, 0.2],
        ).to(device)

        pos_weight = (self.y_train == 0).sum().float() / (self.y_train == 1).sum().float()
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)

        counter = 0
        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            model.train()
            epoch_train_loss = 0
            for cat, num, target in self.train_loader:
                cat, num, target = cat.to(device), num.to(device), target.to(device)
                optimizer.zero_grad()
                loss = criterion(model(cat, num), target.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * len(cat)

            avg_train_loss = epoch_train_loss / len(self.train_loader.dataset)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for cat, num, target in self.test_loader:
                    cat, num, target = cat.to(device), num.to(device), target.to(device)
                    loss = criterion(model(cat, num), target.unsqueeze(1).float())
                    epoch_val_loss += loss.item() * len(cat)

            avg_val_loss = epoch_val_loss / len(self.test_loader.dataset)
            val_losses.append(avg_val_loss)
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping at epoch {epoch}')
                    break

            print(f'Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {scheduler._last_lr[0]:.6f}')

        model.load_state_dict(torch.load(save_path, weights_only=True))
        return model

    def train_lgbm(self, fast_train = True, x_train_cat=None, x_train_num=None, y_train=None):
        model = GradientBoostingDiseaseClassifier('lgbm', fast_train=fast_train)
        model.fit(x_train_cat, x_train_num, y_train, self.categorical_columns, self.numerical_columns)
        return model

    def train_xgb(self, fast_train = True, x_train_cat=None, x_train_num=None, y_train=None):
        model = GradientBoostingDiseaseClassifier('xgb', fast_train=fast_train)
        model.fit(x_train_cat, x_train_num, y_train, self.categorical_columns, self.numerical_columns)
        return model

    def combined_predict(self, mlp_model, lgbm_model, xgb_model, x_test_cat, x_test_num):
        dummy = torch.zeros(x_test_cat.shape[0], dtype=torch.long)
        loader = DataLoader(TensorDataset(x_test_cat, x_test_num, dummy), batch_size=8192, shuffle=False)
        mlp_probs  = mlp_model.predict_proba(loader).numpy()
        lgbm_probs = lgbm_model.predict_proba(x_test_cat, x_test_num, self.categorical_columns, self.numerical_columns)
        xgb_probs  = xgb_model.predict_proba(x_test_cat, x_test_num, self.categorical_columns, self.numerical_columns)
        return (mlp_probs + lgbm_probs + xgb_probs) / 3

    def evaluate_combined(self, threshold = 0.4, mlp_model=None, lgbm_model=None, xgb_model=None, x_test_cat=None, x_test_num=None, thresholds=None):
        if thresholds is None:
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        dummy = torch.zeros(x_test_cat.shape[0], dtype=torch.long)
        loader = DataLoader(TensorDataset(x_test_cat, x_test_num, dummy), batch_size=8192, shuffle=False)
        mlp_probs  = mlp_model.predict_proba(loader).numpy()
        lgbm_probs = lgbm_model.predict_proba(x_test_cat, x_test_num, self.categorical_columns, self.numerical_columns)
        xgb_probs  = xgb_model.predict_proba(x_test_cat, x_test_num, self.categorical_columns, self.numerical_columns)
        combined_probs = (mlp_probs + lgbm_probs + xgb_probs) / 3

        y_true = self.y_test.cpu().numpy()

        print(f'{"Threshold":>10} | {"Accuracy":>10} | {"Precision":>10} | {"Recall":>10} | {"F1":>10}')
        print('-' * 60)
        for t in thresholds:
            preds = (combined_probs > t).astype(int)
            print(
                f'{t:>10.2f} | {(preds == y_true).mean():>10.4f} | '
                f'{precision_score(y_true, preds):>10.4f} | '
                f'{recall_score(y_true, preds):>10.4f} | '
                f'{f1_score(y_true, preds):>10.4f}'
            )

        mlp_preds  = (mlp_probs  > threshold).astype(int)
        lgbm_preds = (lgbm_probs > threshold).astype(int)
        xgb_preds  = (xgb_probs  > threshold).astype(int)

        mlp_TPweight  = float(((mlp_preds  == 1) & (y_true == 1)).sum() / (y_true == 1).sum())
        lgbm_TPweight = float(((lgbm_preds == 1) & (y_true == 1)).sum() / (y_true == 1).sum())
        xgb_TPweight  = float(((xgb_preds  == 1) & (y_true == 1)).sum() / (y_true == 1).sum())

        self.tp_weights = {'mlp': mlp_TPweight, 'lgbm': lgbm_TPweight, 'xgb': xgb_TPweight}

        print(f"MLP TP: {mlp_TPweight:.2%}")
        print(f"LGBM TP: {lgbm_TPweight:.2%}")
        print(f"XGB TP: {xgb_TPweight:.2%}")

        weighted_confidence = (mlp_probs * mlp_TPweight + lgbm_probs * lgbm_TPweight + xgb_probs * xgb_TPweight) / (mlp_TPweight + lgbm_TPweight + xgb_TPweight)

        confirmation_df = pd.DataFrame({
            'actual':              y_true,
            'mlp_pred':            mlp_preds,
            'mlp_confidence':      mlp_probs,
            'lgbm_pred':           lgbm_preds,
            'lgbm_confidence':     lgbm_probs,
            'xgb_pred':            xgb_preds,
            'xgb_confidence':      xgb_probs,
            'unanimous':           (mlp_preds == 1) & (lgbm_preds == 1) & (xgb_preds == 1),
            'weighted_confidence': weighted_confidence,
            'weighted_positive':   weighted_confidence > threshold,
        })
        return confirmation_df

    def predict(self, patient_dict, mlp_model, lgbm_model, xgb_model, threshold=0.4):
        """Predict disease classification for a single patient from raw data.

        Args:
            patient_dict: dict with patient data (human-friendly labels accepted).
                See DataProcessor.process_patient() for expected keys.
            mlp_model: trained ArbovirosesMLP model
            lgbm_model: trained GradientBoostingDiseaseClassifier (lgbm)
            xgb_model: trained GradientBoostingDiseaseClassifier (xgb)
            threshold: classification threshold (default 0.4)

        Returns:
            dict with individual and combined predictions, probabilities, and confidence.
        """
        row = self.data_processor.process_patient(patient_dict)

        cat_cols = list(self.categorical_columns)
        num_cols = [c for c in self.numerical_columns if c in row.columns]

        x_cat = torch.tensor(row[cat_cols].values, dtype=torch.long).to(device)
        x_num = torch.tensor(row[num_cols].values, dtype=torch.float32).to(device)

        # MLP prediction
        dummy = torch.zeros(x_cat.shape[0], dtype=torch.long).to(device)
        loader = DataLoader(TensorDataset(x_cat, x_num, dummy), batch_size=1, shuffle=False)
        mlp_prob = float(mlp_model.predict_proba(loader).numpy()[0])

        # LGBM & XGB predictions
        lgbm_prob = float(lgbm_model.predict_proba(x_cat, x_num, cat_cols, num_cols)[0])
        xgb_prob = float(xgb_model.predict_proba(x_cat, x_num, cat_cols, num_cols)[0])

        # Simple average
        avg_prob = (mlp_prob + lgbm_prob + xgb_prob) / 3

        # Weighted confidence using TP weights (if available from evaluate_combined)
        tp = getattr(self, 'tp_weights', None)
        if tp:
            total = tp['mlp'] + tp['lgbm'] + tp['xgb']
            weighted_prob = (mlp_prob * tp['mlp'] + lgbm_prob * tp['lgbm'] + xgb_prob * tp['xgb']) / total
        else:
            weighted_prob = avg_prob

        return {
            'mlp_probability': mlp_prob,
            'lgbm_probability': lgbm_prob,
            'xgb_probability': xgb_prob,
            'mlp_prediction': int(mlp_prob > threshold),
            'lgbm_prediction': int(lgbm_prob > threshold),
            'xgb_prediction': int(xgb_prob > threshold),
            'average_probability': avg_prob,
            'weighted_probability': weighted_prob,
            'final_prediction': int(weighted_prob > threshold),
            'unanimous': all(p > threshold for p in [mlp_prob, lgbm_prob, xgb_prob]),
        }

    def save_artifacts(self, directory, mlp_model, lgbm_model, xgb_model, embedding_sizes,
                       x_test_cat=None, x_test_num=None, y_test=None, n_examples=200):
        """Save all models and preprocessing artifacts for inference without training data."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save PyTorch MLP
        torch.save(mlp_model.state_dict(), directory / 'mlp.pth')

        # Save LGBM and XGB models
        joblib.dump(lgbm_model.model, directory / 'lgbm_model.joblib')
        joblib.dump(xgb_model.model, directory / 'xgb_model.joblib')

        # Save fitted encoder
        joblib.dump(self.data_processor.encoder, directory / 'encoder.joblib')

        # Save test examples for demo
        if x_test_cat is not None and x_test_num is not None and y_test is not None:
            n = min(n_examples, len(y_test))
            idx = np.random.choice(len(y_test), n, replace=False)
            np.savez(
                directory / 'test_examples.npz',
                x_cat=x_test_cat[idx].cpu().numpy(),
                x_num=x_test_num[idx].cpu().numpy(),
                y=y_test[idx].cpu().numpy(),
            )
            print(f'Saved {n} test examples')

        # Save metadata as JSON
        artifacts = {
            'categorical_columns': list(self.categorical_columns),
            'numerical_columns': list(self.numerical_columns),
            'dropped_columns': self.data_processor.dropped_columns or [],
            'embedding_sizes': [[int(a), int(b)] for a, b in embedding_sizes],
            'age_median': float(self.data_processor.age_median),
            'health_region_median': int(self.data_processor.health_region_median),
            'days_to_notification_median': float(self.data_processor.days_to_notification_median),
            'tp_weights': getattr(self, 'tp_weights', None),
            'type_disease': self.data_processor.type_disease,
        }
        with open(directory / 'artifacts.json', 'w') as f:
            json.dump(artifacts, f, indent=2)

        print(f'Artifacts saved to {directory}')

    @staticmethod
    def load_for_inference(directory):
        """Load models and artifacts from disk for inference (no training data needed)."""
        directory = Path(directory)

        with open(directory / 'artifacts.json') as f:
            artifacts = json.load(f)

        # Build a lightweight orchestrator without loading training CSVs
        orch = object.__new__(ModelsOrchestrator)
        orch.categorical_columns = artifacts['categorical_columns']
        orch.numerical_columns = artifacts['numerical_columns']
        orch.tp_weights = artifacts['tp_weights']
        orch.train_loader = None
        orch.test_loader = None
        orch.y_train = None
        orch.y_test = None

        # Restore DataProcessor with saved artifacts (no CSV loading)
        dp = object.__new__(DataProcessor)
        dp.type_disease = artifacts['type_disease']
        dp.encoder = joblib.load(directory / 'encoder.joblib')
        dp.dropped_columns = artifacts['dropped_columns']
        dp.age_median = artifacts['age_median']
        dp.health_region_median = artifacts['health_region_median']
        dp.days_to_notification_median = artifacts['days_to_notification_median']
        orch.data_processor = dp

        # Load MLP
        embedding_sizes = [tuple(e) for e in artifacts['embedding_sizes']]
        mlp_model = ArbovirosesMLP(
            numericals_shape=len(artifacts['numerical_columns']),
            embedding_sizes=embedding_sizes,
            hidden_layers=[1024, 512, 256, 128],
            probability_dropout=[0.1, 0.2],
        ).to(device)
        mlp_model.load_state_dict(torch.load(directory / 'mlp.pth', weights_only=True, map_location=device))
        mlp_model.eval()

        # Load LGBM
        lgbm_model = object.__new__(GradientBoostingDiseaseClassifier)
        lgbm_model.model = joblib.load(directory / 'lgbm_model.joblib')
        lgbm_model.feature_names = artifacts['categorical_columns'] + list(artifacts['numerical_columns'])
        lgbm_model._model_type = 'lgbm'

        # Load XGB
        xgb_model = object.__new__(GradientBoostingDiseaseClassifier)
        xgb_model.model = joblib.load(directory / 'xgb_model.joblib')
        xgb_model.feature_names = artifacts['categorical_columns'] + list(artifacts['numerical_columns'])
        xgb_model._model_type = 'xgb'

        # Load test examples if available
        test_examples = None
        npz_path = directory / 'test_examples.npz'
        if npz_path.exists():
            data = np.load(npz_path)
            test_examples = {
                'x_cat': torch.tensor(data['x_cat'], dtype=torch.long),
                'x_num': torch.tensor(data['x_num'], dtype=torch.float32),
                'y': data['y'],
            }

        return orch, mlp_model, lgbm_model, xgb_model, test_examples
