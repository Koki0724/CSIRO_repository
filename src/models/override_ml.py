# src/models/wrappers.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

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

class LGBMWrapper(BaseModelWrapper):
    def __init__(self, params=None, num_boost_round=1000, early_stopping_rounds=50, verbose_eval=100, **kwargs):
        """
        params: lightgbm.yaml から渡されるハイパーパラメータ辞書
        """
        self.params = params if params is not None else {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.model = None
        self.encoders = {} # カテゴリ変数のエンコーダー保存用

    def _process_features(self, df, is_train=True):
        """
        テーブルデータの特徴量エンジニアリング
        """
        df = df.copy()
        
        if 'Sampling_Date' in df.columns:
            df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])
            df['year'] = df['Sampling_Date'].dt.year
            df['month'] = df['Sampling_Date'].dt.month
            df['day_of_year'] = df['Sampling_Date'].dt.dayofyear
            df = df.drop(columns=['Sampling_Date'])

        cat_cols = ['State', 'Species']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Missing").astype(str)
                
                if is_train:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                else:
                    le = self.encoders.get(col)
                    if le:
                        df[col] = df[col].map(lambda x: x if x in le.classes_ else le.classes_[0])
                        df[col] = le.transform(df[col])

        use_cols = ['year', 'month', 'day_of_year', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm']
        
        return df[use_cols]

    def _concat_embed_and_features(self, X, df_processed):
        """
        画像Embedding(numpy)とテーブル特徴量(DataFrame)を結合する
        """
        
        X_df = pd.DataFrame(X, columns=[f"embed_{i}" for i in range(X.shape[1])])
        
        # テーブルデータ側もindexをリセットして横に結合
        df_reset = df_processed.reset_index(drop=True)
        
        combined_df = pd.concat([X_df, df_reset], axis=1)
        return combined_df

    def fit_predict(self, X_train, y_train, X_val, y_val, df_train, df_val):
        
        # 1. テーブルデータの前処理
        df_train_proc = self._process_features(df_train, is_train=True)
        df_val_proc = self._process_features(df_val, is_train=False)
        
        # 2. 画像とテーブルの結合
        X_train_all = self._concat_embed_and_features(X_train, df_train_proc)
        X_val_all = self._concat_embed_and_features(X_val, df_val_proc)
        
        # 3. LightGBMデータセット作成
        lgb_train = lgb.Dataset(X_train_all, y_train)
        lgb_val = lgb.Dataset(X_val_all, y_val, reference=lgb_train)
        
        # 4. 学習
        self.model = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
                lgb.log_evaluation(self.verbose_eval)
            ]
        )
        
        preds = self.model.predict(X_val_all, num_iteration=self.model.best_iteration)
        
        return preds, self

    def predict(self, X_test, df_test=None):
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        
        if df_test is None:
            raise ValueError("LightGBM Wrapper requires df_test for tabular features.")
            
        # テストデータの前処理
        df_test_proc = self._process_features(df_test, is_train=False)
        
        # 結合
        X_test_all = self._concat_embed_and_features(X_test, df_test_proc)
        
        return self.model.predict(X_test_all, num_iteration=self.model.best_iteration)
