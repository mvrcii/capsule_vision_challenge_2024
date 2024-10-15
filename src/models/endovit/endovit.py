import logging
from functools import partial
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from src.models.abstract_model import AbstractLightningModule


class EndoViT(AbstractLightningModule):
    def __init__(self, config, class_to_idx, checkpoint_path=None):
        super().__init__(config, class_to_idx, checkpoint_path)

    def init_backbone(self):
        model_path = snapshot_download(repo_id=f"egeozsoy/{self.model_arch}", revision="main")
        model_weights_path = Path(model_path) / "pytorch_model.bin"

        logging.info(f"Model: Backbone {self.model_type} with weights {self.model_arch} built.")

        # Load model weights
        model_weights = torch.load(model_weights_path)['model']

        # Define the model (ensure this matches your model's architecture)
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # Load the weights into the model
        model.load_state_dict(model_weights, strict=False)
        model.head = nn.Identity()
        return model
