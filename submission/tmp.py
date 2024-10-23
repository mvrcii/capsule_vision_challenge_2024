from transformers import AutoConfig

config = AutoConfig.from_pretrained("timm/eva02_base_patch14_224.mim_in22k")

print(config)