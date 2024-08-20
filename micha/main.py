from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the directory paths
train_dir = '../data/training'
val_dir = '../data/validation'

# Define the image transformations
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation: horizontal flip
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization
    ]),
}

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=image_transforms['train'])
val_dataset = datasets.ImageFolder(root=val_dir, transform=image_transforms['val'])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Get the class names
class_names = train_dataset.classes

print("Classes:", class_names)
