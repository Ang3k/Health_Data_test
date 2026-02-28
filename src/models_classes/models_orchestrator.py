import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from data_processing.disease_dataset_process import DataProcessor
from models_classes.lgbm_classifier import LGBMDiseaseClassifier
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
        self.train_loader = DataLoader(TensorDataset(x_train_cat, x_train_num, y_train), batch_size=4096, shuffle=True)
        self.test_loader  = DataLoader(TensorDataset(x_test_cat,  x_test_num,  y_test),  batch_size=4096, shuffle=False)

        return x_train_cat, x_test_cat, x_train_num, x_test_num, y_train, y_test, embedding_sizes

    def train_mlp(self, embedding_sizes, save_path, epochs=150, patience=8):
        model = ArbovirosesMLP(
            numericals_shape=len(self.numerical_columns),
            embedding_sizes=embedding_sizes,
            hidden_layers=[2048, 1024, 512, 256],
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

    def train_lgbm(self, x_train_cat, x_train_num, y_train):
        model = LGBMDiseaseClassifier()
        model.fit(x_train_cat, x_train_num, y_train, self.categorical_columns, self.numerical_columns)
        return model

    def combined_predict(self, mlp_model, lgbm_model, x_test_cat, x_test_num):
        dummy = torch.zeros(x_test_cat.shape[0], dtype=torch.long)
        loader = DataLoader(TensorDataset(x_test_cat, x_test_num, dummy), batch_size=4096, shuffle=False)
        mlp_probs  = mlp_model.predict_proba(loader).numpy()
        lgbm_probs = lgbm_model.predict_proba(x_test_cat, x_test_num, self.categorical_columns, self.numerical_columns)
        return (mlp_probs + lgbm_probs) / 2

    def evaluate_combined(self, mlp_model, lgbm_model, x_test_cat, x_test_num, thresholds=None):
        if thresholds is None:
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        combined_probs = self.combined_predict(mlp_model, lgbm_model, x_test_cat, x_test_num)
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

        preds_04 = (combined_probs > 0.4).astype(int)
        confirmation_df = pd.DataFrame({
            'xgb':        y_true,
            'neural_net': preds_04,
            'confirmation': y_true == preds_04,
        })
        return confirmation_df
