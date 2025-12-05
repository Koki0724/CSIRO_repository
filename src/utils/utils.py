import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold

def calc_metrics(y_true, y_pred):
    """
    RMSEとR2スコアを計算して返す
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

def calc_weighted_metrics(y_true, y_pred, target_weights):
    """
    画像の数式に基づき、全要素を一括として重み付きRMSEとR2を計算する
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    target_weights = np.array(target_weights)
    
    # 形状合わせ: (5, N) -> (5, N) の重み行列を作成
    if y_true.shape != y_pred.shape:
         raise ValueError(f"Shape mismatch: y_true {y_true.shape}, y_pred {y_pred.shape}")
         
    W = np.tile(target_weights[:, np.newaxis], (1, y_true.shape[1]))
        
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    W_flat = W.ravel()
    
    mse = mean_squared_error(y_true_flat, y_pred_flat, sample_weight=W_flat)
    rmse = np.sqrt(mse)
    
    r2 = r2_score(y_true_flat, y_pred_flat, sample_weight=W_flat)
    
    return rmse, r2

def preprocess_train_data(train_df, n_splits=5):
    """
    1. 画像ごとに1行になるようにピボット（データの整列問題を解決）
    2. Group(Sampling_Date) かつ Stratified(State) でFoldを作成
    """
    # ターゲット名をカラムに変換 (Wide形式へ)
    # これにより image_path と target の対応関係が完全に保証されます
    pivot_df = train_df.pivot_table(
        index=['image_path', 'Sampling_Date', 'State'], 
        columns='target_name', 
        values='target'
    ).reset_index()

    # Foldの作成
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    pivot_df['fold'] = -1
    
    # y=State (層化), groups=Sampling_Date (グループ)
    # StratifiedGroupKFold は y を層化の基準、groups をグループの基準とします
    for fold, (_, val_idx) in enumerate(sgkf.split(pivot_df, pivot_df['State'], groups=pivot_df['Sampling_Date'])):
        pivot_df.loc[val_idx, 'fold'] = fold
        
    return pivot_df