import logging

import timm

from src.models.abstract_model import AbstractLightningModule


class RegNetY(AbstractLightningModule):
    def __init__(self, config, class_to_idx, checkpoint_path=None):
        super().__init__(config, class_to_idx, checkpoint_path)

    def init_backbone(self):
        if self.model_arch == "regnety_640":
            logging.info(f"Model: Building {self.model_arch} with random initialized weights.")
            return timm.create_model(self.model_arch, pretrained=False, num_classes=0)
        else:
            logging.info(f"Model: Backbone {self.model_type} with weights {self.model_arch} built.")
            return timm.create_model(
                model_name=self.model_arch,
                pretrained=not self.checkpoint_path,  # If checkpoint is provided, do not load pretrained weights
                num_classes=0  # No head/classifier
            )
