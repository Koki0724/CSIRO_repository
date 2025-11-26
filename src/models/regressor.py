# src/models/regressor.py
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

class LassoEnsemble:
    def __init__(self, n_splits: int = 5, random_state: int = 42, alpha: float = 1.0):
        self.n_splits = n_splits
        self.random_state = random_state
        self.alpha = alpha
        self.models = [] 

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.models = [] 
        oof_preds = np.zeros(len(X))
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val = X[val_idx]
            
            model = Lasso(alpha=self.alpha)
            model.fit(X_train, y_train)
            
            oof_preds[val_idx] = model.predict(X_val)
            self.models.append(model)
            
        return oof_preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model not fitted yet!")
        
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return np.mean(preds, axis=0)