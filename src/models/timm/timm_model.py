import logging

import timm
import torch

from src.models.abstract_model import AbstractLightningModule


class TimmModel(AbstractLightningModule):
    def __init__(self, config, class_to_idx, checkpoint_path=None):
        super().__init__(config, class_to_idx, checkpoint_path)

    def init_backbone(self):
        logging.info(f"Model {self.config.model_type} with weights {self.config.model_arch} built.")
        model = timm.create_model(
            model_name=self.config.model_arch,
            pretrained=not self.checkpoint_path,  # If checkpoint is provided, do not load pretrained weights
            num_classes=0  # No head/classifier
        )

        # Load from checkpoint if path is provided
        if self.checkpoint_path:
            logging.info(f"Loading weights from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])

        return model