import torch.nn as nn
import timm

class TimmModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, num_classes: int = 1):
        """
        簡単な画像モデルのラッパークラス
        Args:
            model_name (str): モデル名 (例: 'resnet18')
            pretrained (bool): 学習済み重みを使うか
            num_classes (int): 出力の数 (回帰なら1)
        """
        super().__init__()
        
        # timmを使ってモデルをロード
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )

    def forward(self, x):
        # 順伝播 (データxを入力して予測値を出す)
        return self.model(x)