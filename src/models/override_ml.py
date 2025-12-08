# src/models/wrappers.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline

class BaseModelWrapper:
    """
    全てのモデルラッパーの基底クラス
    run_ml.pyからはこのインターフェースを通じて呼び出される
    """
    def fit_predict(self, X_train, y_train, X_val, y_val, df_train, df_val):
        raise NotImplementedError

    def predict(self, X_test, df_test=None):
        raise NotImplementedError

class LassoWrapper(BaseModelWrapper):
    def __init__(self, alpha=1.0, normalization="standard", random_state=42, **kwargs):
        self.alpha = alpha
        self.normalization = normalization
        self.random_state = random_state
        self.model = None

    def fit_predict(self, X_train, y_train, X_val, y_val, df_train, df_val):
        """
        学習を行い、Validationデータに対する予測値と学習済みモデルを返す
        """
        # 1. モデルの構築 (Pipeline)
        base_model = Lasso(alpha=self.alpha, random_state=self.random_state)
        
        # 正規化の切り替えロジックをここに集約
        if self.normalization == "standard":
            self.model = make_pipeline(StandardScaler(), base_model)
        elif self.normalization == "l2":
            self.model = make_pipeline(Normalizer(norm='l2'), base_model)
        else:
            self.model = base_model


        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_val).flatten()
        
        return preds, self

    def predict(self, X_test, df_test=None):
        """
        テストデータの予測
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        return self.model.predict(X_test).flatten()

