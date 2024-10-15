import albumentations as A
from albumentations.pytorch import ToTensorV2
img_size = 42

train_transforms = A.Compose([
    A.Resize(height=img_size, width=img_size, interpolation=2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
val_transforms = A.Compose([
    A.Resize(height=img_size, width=img_size, interpolation=2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(always_apply=True)
])
