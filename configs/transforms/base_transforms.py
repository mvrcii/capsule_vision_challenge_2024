import albumentations as A
from albumentations.pytorch import ToTensorV2
img_size = 42

train_transforms = A.Compose([
    A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333),
                        interpolation=2),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(p=0.5, brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
val_transforms = A.Compose([
    A.Resize(height=img_size, width=img_size, interpolation=2),
    A.CenterCrop(height=img_size, width=img_size, always_apply=True),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(always_apply=True)
])
