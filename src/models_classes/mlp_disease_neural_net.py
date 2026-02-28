import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.inspection import permutation_importance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DengueTabularNN(nn.Module):
    def __init__(self, numericals_shape, embedding_sizes, hidden_layers=[600, 300, 200, 100], probability_dropout=[0.05, 0.3]):
        super().__init__()

        lista_embeddings = [nn.Embedding(size, new_size) for size, new_size in embedding_sizes]
        self.embeddings = nn.ModuleList(lista_embeddings)
        self.dropout_embeddings = nn.Dropout(p=probability_dropout[0])

        self.normalization = nn.BatchNorm1d(numericals_shape)

        sum_columns = sum([embedding_sizes[i][1] for i in range(len(embedding_sizes))]) + numericals_shape

        layers = []
        current_entries = sum_columns
        for layer_neurons in hidden_layers:
            layers.append(nn.Linear(current_entries, layer_neurons))
            layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(layer_neurons))
            layers.append(nn.Dropout(p=probability_dropout[1]))
            current_entries = layer_neurons

        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_entries, 1)

    @staticmethod
    def _prepare_data(df, categorical_columns):
        categorical_tensors = torch.tensor(df[categorical_columns].values, dtype=torch.long).to(device)
        numerical_tensors = torch.tensor(df.drop(columns=categorical_columns + ['final_classification']).values, dtype=torch.float).to(device)
        target_tensor = torch.tensor(df['final_classification'].values, dtype=torch.long).to(device)

        unique_cat = [df[col].max() + 1 for col in categorical_columns]
        embedding_sizes = [(size, min(50, (size // 2) + 1)) for size in unique_cat]

        return categorical_tensors, numerical_tensors, target_tensor, embedding_sizes

    def _engineering_embeddings(self, x_categorical):
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(x_categorical[:, i]))
        return torch.cat(embedded, dim=1)

    def forward(self, x_categorical, x_numerical):
        x_categorical = self._engineering_embeddings(x_categorical)
        x_categorical = self.dropout_embeddings(x_categorical)
        x_numerical = self.normalization(x_numerical)
        x = torch.cat([x_categorical, x_numerical], dim=1)
        x = self.layers(x)
        return self.output_layer(x)

    def predict_proba(self, test_loader):
        self.eval()
        all_probs = []
        with torch.no_grad():
            for x_cat, x_num, _ in test_loader:
                out = self(x_cat, x_num)
                all_probs.append(torch.sigmoid(out).squeeze().cpu())
        return torch.cat(all_probs)

    def evaluate(self, test_loader, y_test, thresholds=None):
        if thresholds is None:
            thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        probabilities = self.predict_proba(test_loader)
        y_true = y_test.cpu()

        print(f'{"Threshold":>10} | {"Accuracy":>10} | {"Precision":>10} | {"Recall":>10} | {"F1":>10}')
        print('-' * 60)
        for t in thresholds:
            preds = (probabilities > t).long()
            print(
                f'{t:>10.2f} | {(preds == y_true).float().mean().item():>10.4f} | '
                f'{precision_score(y_true, preds):>10.4f} | '
                f'{recall_score(y_true, preds):>10.4f} | '
                f'{f1_score(y_true, preds):>10.4f}'
            )

        predicted_classes = (probabilities > 0.3).long()
        return pd.DataFrame({
            'Actual': y_true.numpy(),
            'Prob': probabilities.numpy(),
            'Predicted': predicted_classes.numpy(),
            'Correct': (y_true == predicted_classes).numpy(),
        })

    def plot_feature_importance(self, x_test_cat, x_test_num, y_test, categorical_columns, numerical_columns, top_n=20, sample_size=2000, n_repeats=100):
        idx = np.random.choice(x_test_cat.shape[0], size=sample_size, replace=False)

        X_test = np.concatenate([
            x_test_cat[idx].cpu().numpy(),
            x_test_num[idx].cpu().numpy().astype(np.float32),
        ], axis=1)
        y_test_np = y_test[idx].cpu().numpy().astype(int).flatten()

        n_cat = x_test_cat.shape[1]
        model = self

        class _SklearnWrapper:
            def fit(self, *_): return self
            def predict(self, X):
                cat = torch.tensor(X[:, :n_cat], dtype=torch.long).to(device)
                num = torch.tensor(X[:, n_cat:], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out = model(cat, num)
                return (out.cpu().numpy() > 0.5).astype(int).flatten()
            def score(self, X, y):
                return accuracy_score(y, self.predict(X))

        result = permutation_importance(_SklearnWrapper(), X_test, y_test_np, n_repeats=n_repeats, random_state=42)

        all_feature_names = list(categorical_columns) + list(numerical_columns)
        sorted_idx = result.importances_mean.argsort()[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.bar(range(top_n), result.importances_mean[sorted_idx], yerr=result.importances_std[sorted_idx])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.xticks(range(top_n), [all_feature_names[i] for i in sorted_idx], rotation=90)
        plt.title(f'Permutation Feature Importance - Top {top_n}', fontsize=14)
        plt.tight_layout()
        plt.show()
