import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier as _LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score


class GradientBoostingDiseaseClassifier:
    def __init__(self, model, n_estimators=2000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, device='gpu'):

        if model == 'lgbm':
            self.model = _LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                device=device,
            )
            self.feature_names = None

        elif model == 'xgb':
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree
            )
            self.feature_names = None

    def _prepare_data(self, x_cat, x_num, categorical_columns, numerical_columns):
        return pd.concat([
            pd.DataFrame(x_cat.cpu().numpy(), columns=categorical_columns),
            pd.DataFrame(x_num.cpu().numpy(), columns=numerical_columns),
        ], axis=1)

    def fit(self, x_train_cat, x_train_num, y_train, categorical_columns, numerical_columns):
        self.feature_names = categorical_columns + list(numerical_columns)
        X_train = self._prepare_data(x_train_cat, x_train_num, categorical_columns, numerical_columns)
        self.model.fit(X_train, y_train.cpu().numpy())

    def predict_proba(self, x_cat, x_num, categorical_columns, numerical_columns):
        X = self._prepare_data(x_cat, x_num, categorical_columns, numerical_columns)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, x_test_cat, x_test_num, y_test, categorical_columns, numerical_columns, thresholds=None):
        if thresholds is None:
            thresholds = [0.1, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        probs = self.predict_proba(x_test_cat, x_test_num, categorical_columns, numerical_columns)
        y_true = y_test.cpu().numpy()

        print(f'{"Threshold":>10} | {"Accuracy":>10} | {"Precision":>10} | {"Recall":>10} | {"F1":>10}')
        print('-' * 60)
        for t in thresholds:
            preds = (probs > t).astype(int)
            print(
                f'{t:>10.2f} | {(preds == y_true).mean():>10.4f} | '
                f'{precision_score(y_true, preds):>10.4f} | '
                f'{recall_score(y_true, preds):>10.4f} | '
                f'{f1_score(y_true, preds):>10.4f}'
            )

    def plot_feature_importance(self, top_n=30):
        importances = pd.Series(self.model.feature_importances_, index=self.feature_names)
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        importances.head(top_n).plot(kind='bar')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.title(f'LightGBM - Top {top_n} Feature Importances')
        plt.ylabel('Importance (F-score)')
        plt.tight_layout()
        plt.show()

        print(importances.to_string())
