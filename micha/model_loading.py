import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTForImageClassification

import wandb  # Import wandb

# Define the directory paths
train_dir = '../data/training'
val_dir = '../data/validation'

# Define the transform to apply to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (size expected by ViT)
    transforms.ToTensor(),  # Convert the image to a tensor
])

# Load datasets with the transform
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# train subset
dataset_size = len(train_dataset)
count = int(dataset_size * 0.5)
indices = np.random.permutation(dataset_size)[:count]  # Get 20% of indices
train_subset = Subset(train_dataset, indices)

# val subset
validation_size = len(val_dataset)
count = int(validation_size * 0.25)
indices_val = np.random.permutation(validation_size)[:count]  # Get 25% of indices
validation_subset = Subset(val_dataset, indices_val)

# Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=256, shuffle=True, num_workers=1)
val_loader = DataLoader(validation_subset, batch_size=256, shuffle=False, num_workers=1)

# Get the class names
class_names = train_dataset.classes


def train_and_evaluate(num_epochs=20):
    # Initialize W&B project
    wandb.init(project='CV2024', entity='wuesuv', config={
        'num_epochs': num_epochs,
        'batch_size': 256,
        'learning_rate': 5e-5,
    })

    # Load the image processor and the model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Modify the classifier head for 10 classes
    model.classifier = nn.Linear(in_features=768, out_features=10, bias=True)

    # Freeze all model parameters except the classifier head
    for param in model.parameters():
        param.requires_grad = False

    # Ensure the classifier head parameters are trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Send model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define the Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Process images with the ViT image processor
            inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)

            # Forward pass
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Log training loss to W&B
        wandb.log({'Training Loss': running_loss / len(train_loader)})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        if epoch % 5 == 0:
            # Validation Loop
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images = images.to(device)
                    labels = labels.to(device)

                    # Process images with the ViT image processor
                    inputs = processor(images=images, return_tensors="pt", do_rescale=False).to(device)

                    outputs = model(**inputs)
                    _, predicted = torch.max(outputs.logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Calculate accuracy
            accuracy = correct / total
            # Log validation accuracy to W&B
            wandb.log({'Validation Accuracy': accuracy * 100})

            print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # Finish W&B run
    wandb.finish()


if __name__ == '__main__':
    train_and_evaluate()
