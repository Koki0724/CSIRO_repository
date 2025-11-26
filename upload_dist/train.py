import sys
import os
import hydra
from omegaconf import DictConfig
import torch

# 【重要】親ディレクトリ(ルート)をパスに追加して、srcフォルダを読み込めるようにする
# これがないと "ModuleNotFoundError: No module named 'src'" になります
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"=== 実験開始: {cfg.exp_name} ===")
    print(f"Model Config: {cfg.model}")

    # 1. Hydraを使ってモデルを生成 (ここで src/models/base.py が呼ばれる)
    model = hydra.utils.instantiate(cfg.model)
    print(f"モデル作成成功: {type(model)}")

    # 2. 試しにダミーデータを入れて動くか確認
    # バッチサイズ2, 3チャンネル, 224x224の画像
    dummy_input = torch.randn(2, 3, 224, 224)
    
    try:
        output = model(dummy_input)
        print(f"順伝播成功！ 出力サイズ: {output.shape}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()