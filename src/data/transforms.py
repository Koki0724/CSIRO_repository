# src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class TransformFactory:
    """
    学習および推論（TTA）で使用するTransformを生成するクラス
    """
    def __init__(self, img_size: int = 1000):
        self.img_size = img_size
        
        # ImageNetの平均・分散での正規化とTensor変換（共通処理）
        self.base_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]

    def get_train_transforms(self) -> A.Compose:
        """学習用: ここにAugmentationを追加できます"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # 必要に応じてここにShiftScaleRotateなどを追加
            *self.base_transforms
        ])

    def get_valid_transforms(self) -> A.Compose:
        """検証用: リサイズのみ"""
        return A.Compose([
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

    def get_tta_transforms(self) -> list[A.Compose]:
        """
        推論用TTA: 3つのビュー（オリジナル、左右反転、上下反転）を返す
        """
        # 1. Original
        original = A.Compose([
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # 2. Horizontal Flip
        hflip = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        # 3. Vertical Flip
        vflip = A.Compose([
            A.VerticalFlip(p=1.0),
            A.Resize(self.img_size, self.img_size),
            *self.base_transforms
        ])

        return [original, hflip, vflip]

class BaseTransform:
    """
    Transformの基底クラス
    __call__ を実装することで、インスタンスを関数のように呼べるようにする
    """
    def __init__(self):
        self.transform = A.Compose([])

    def __call__(self, **kwargs):
        # Dataset内で transform(image=img) と呼ばれたときに動く
        return self.transform(**kwargs)

class TrainTransform(BaseTransform):
    def __init__(self, img_size: int = 224, aug_prob: float = 0.5):
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # 幾何学的変換
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.1, 
                rotate_limit=30, 
                border_mode=cv2.BORDER_REFLECT, 
                p=aug_prob
            ),
            
            # 色調変化
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            ], p=aug_prob),
            
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

class ValidTransform(BaseTransform):
    def __init__(self, img_size: int = 224):
        self.transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

class InferenceTransform(BaseTransform):
    def __init__(self, img_size: int = 224, tta_idx: int = 0):
        """
        tta_idx: 0=Original, 1=HorizontalFlip, 2=VerticalFlip
        """
        transforms = [A.Resize(img_size, img_size)]
        
        # TTAロジック
        if tta_idx == 1:
            transforms.append(A.HorizontalFlip(p=1.0))
        elif tta_idx == 2:
            transforms.append(A.VerticalFlip(p=1.0))
            
        transforms.extend([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        self.transform = A.Compose(transforms)