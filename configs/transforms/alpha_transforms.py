import albumentations as A
from albumentations.pytorch import ToTensorV2

img_size = 42

train_transforms = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomRotate90(),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=2.55),
    A.RandomResizedCrop(height=img_size, width=img_size, scale=(0.5, 1.0), ratio=(1.0, 1.0), interpolation=1),
    A.Resize(height=img_size, width=img_size, interpolation=4),
    A.GridDistortion(num_steps=3, distort_limit=(-0.09, 0.09), interpolation=0, border_mode=0),
    A.ColorJitter(p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.05, 0.05)),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
val_transforms = A.Compose([
    A.Resize(height=img_size, width=img_size, interpolation=2),
    A.CenterCrop(height=img_size, width=img_size, always_apply=True),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(always_apply=True)
])
