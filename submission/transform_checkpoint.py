import argparse
import json
import os

import timm
import torch
import wandb
from transformers import PreTrainedModel, AutoConfig

from src.models.timm.timm_model import TimmModel
from src.utils.class_mapping import load_class_mapping

checkpoint_path = "submission/run-20241018_135423-skilled-snow-80/best_epoch38_val_recall_macro0.89.ckpt"

api = wandb.Api()
run = api.run(f'wuesuv/CV2024/rlrkbbvt')
config = argparse.Namespace(**run.config)

config.dataset_csv_path = "../endoscopy-dataset/datasets/cvip2024"
class_mapping_path = os.path.join(config.dataset_csv_path, 'class_mapping.json')
class_mapping = load_class_mapping(class_mapping_path)

huggingface_config_dict = {
    "model_arch": config.model_arch,
    "num_labels": len(class_mapping),
    "learning_rate": config.lr,
    "weight_decay": config.weight_decay,
    "optimizer": config.optimizer,
    "scheduler": config.scheduler,
    "dataset": "cv24"
}

# Save class mapping as JSON
with open("./my_model_dir/class_mapping.json", 'w') as f:
    json.dump(class_mapping, f)

model = TimmModel.load_from_checkpoint(
    checkpoint_path=checkpoint_path,
    config=config,
    class_to_idx=class_mapping
)

# Extract the backbone (from timm)
backbone = model.backbone

# Extract the classifier
classifier = model.classifier

# Save the backbone and classifier as two separate PyTorch models
torch.save(backbone.state_dict(), "backbone_model.pth")
torch.save(classifier.state_dict(), "classifier_model.pth")


class CustomTimmModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Load the timm backbone (eva02 model)
        self.backbone = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=False, num_classes=0)
        self.backbone.load_state_dict(torch.load("backbone_model.pth"))

        # Add the custom classifier (from your fine-tuned Lightning module)
        self.classifier = torch.nn.Linear(self.backbone.num_features, config.num_labels)
        self.classifier.load_state_dict(torch.load("classifier_model.pth"))

    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        # Pass through the custom classifier head
        logits = self.classifier(features)
        return logits


config = AutoConfig.from_pretrained("timm/eva02_base_patch14_224.mim_in22k")

# Initialize the model with the config
model = CustomTimmModel(config)

# Save the model in Hugging Face format
model.save_pretrained("./my_model_dir")
config.save_pretrained("./my_model_dir")
