import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier as _LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import optuna
from sklearn.model_selection import cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)


class GradientBoostingDiseaseClassifier:
    def __init__(self, model, fast_train=True, n_estimators=2000, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, device='gpu'):
        self._model_type = model
        self._fast_train = fast_train
        self._device = device
        self.feature_names = None

        if fast_train:
            if model == 'lgbm':
                self.model = _LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    device=device,
                )

            elif model == 'xgb':
                self.model = XGBClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    device=device,
                )
        else:
            self.model = None  # set after Optuna optimization in fit()

    def _optimize(self, X, y, n_trials=50):
        model_type = self._model_type
        device = self._device

        def objective(trial):
            if model_type == 'lgbm':
                params = {
                    'n_estimators':      trial.suggest_int('n_estimators', 500, 3000, step=500),
                    'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'max_depth':         trial.suggest_int('max_depth', 3, 8),
                    'num_leaves':        trial.suggest_int('num_leaves', 31, 255),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                model = _LGBMClassifier(device=device, verbose=-1, **params)
            else:
                params = {
                    'n_estimators':     trial.suggest_int('n_estimators', 500, 3000, step=500),
                    'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'max_depth':        trial.suggest_int('max_depth', 3, 8),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                model = XGBClassifier(device=device, **params)

            return cross_val_score(model, X, y, cv=3, scoring='recall').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        print(f'Best recall: {study.best_value:.4f}')
        print(f'Best params: {study.best_params}')

        if model_type == 'lgbm':
            return _LGBMClassifier(device=device, verbose=-1, **study.best_params)
        else:
            return XGBClassifier(device=device, **study.best_params)

    def _prepare_data(self, x_cat, x_num, categorical_columns, numerical_columns):
        return pd.concat([
            pd.DataFrame(x_cat.cpu().numpy(), columns=categorical_columns),
            pd.DataFrame(x_num.cpu().numpy(), columns=numerical_columns),
        ], axis=1)

    def fit(self, x_train_cat, x_train_num, y_train, categorical_columns, numerical_columns, n_trials=50):
        self.feature_names = categorical_columns + list(numerical_columns)
        X_train = self._prepare_data(x_train_cat, x_train_num, categorical_columns, numerical_columns)
        y = y_train.cpu().numpy()

        if not self._fast_train:
            self.model = self._optimize(X_train, y, n_trials=n_trials)

        self.model.fit(X_train, y)

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
