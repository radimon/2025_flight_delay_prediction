import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error
import joblib

class LGBMModel:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=-1):
        self.model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.features = None

    def _prepare_features(self, df, seq_len):
        # 根據 preprocess.py 的命名規則
        lag_cols = [f"lag_{k}" for k in range(1, seq_len + 1)]
        static_cols = ["weekday", "t", "is_weekend"]
        self.features = lag_cols + static_cols
        return df[self.features]

    def train(self, train_df, val_df, seq_len):
        X_train = self._prepare_features(train_df, seq_len)
        y_train = np.log1p(train_df["count"])
        
        X_val = self._prepare_features(val_df, seq_len)
        y_val = np.log1p(val_df["count"])

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[None] # 可加入 early_stopping
        )
        return self.model

    def predict(self, df, seq_len):
        X = self._prepare_features(df, seq_len)
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)

def save_lgbm(model, path):
    joblib.dump(model, path)

def load_lgbm(path):
    return joblib.load(path)