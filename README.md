# Capsule Vision Challenge 2024

## Quickstart

1. Create a Virtual Environment with Python 3.11 via ```python3.11 -m venv venv``` and activate
   it ```source venv/bin/activate```
2. Install pytorch 2.4.0+cu121
   via ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```
3. Install required packages with `pip install -r requirements.txt`
4. Download the Capsule Vision Challenge
   2024 [dataset](https://github.com/misahub2023/Capsule-Vision-2024-Challenge?tab=readme-ov-file) from the official
   GitHub repository, store it in `data/` **in the repository root** and make sure to have the following structure:
   ```
    data/
    ├── capsulevision/
    │   ├── training/
    │   │   ├── Angioectasia/
    │   │   │   ├── KID/
    │   │   │   │   ├── angioectasia-P0-1.jpg
    │   │   │   │   ├── ...
    │   │   │   ├── KVASIR/
    │   │   │   │   ├── 04a78ef00c5245e0_888.jpg
    │   │   │   │   ├── ...
    │   │   │   ├── SEE-AI/
    │   │   │   │   ├── image00279.jpg
    │   │   │   │   ├── ...   
    │   ├── validation/
    │   │   ├── ...
    │   ├── testing/
    │   │   ├── Images
    │   │   │   ├── 00Z0Xo99wp.jpg
    │   │   │   ├── ...
   ```
5. Download the pre-trained EndoExtend24 model weights via link/command and store them in `pretrained_models/` **in the repository root**
6. Make sure that your **repository root** is structured as follows:
```
.
├── configs/
├── data/
├── datasets/
│   ├── ce24/
│   │   ├── class_mapping.json
│   │   ├── train_val.csv
│   │   ├── test.csv
├── pretrained_models/
│   ├── eva02_base_patch14_224.pt_ee24.ckpt
│   ├── eva02_base_patch14_224.ee24_ft_ce24.ckpt   
├── src/
├── README.md
├── requirements.txt
├── infer.py
├── train.py
├── ...
```
7. Run `python train.py --config configs/submission/run_eva02_base_patch14_224.ee24_ft_ce24.yaml` to start fine-tuning the pre-trained EndoExtend24 model.

## Model Weights

You can find the weights for both the pre-trained model and the fine-tuned downstream task model below. For an easy
download on a bash cluster such as a slurm master, you can use the `gdown` command as shown below.

| **Type**    | **Dataset**  | **Checkpoint**                                                                                                            | **Command**                               |
|-------------|--------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| Pre-trained | EndoExtend24 | [eva02_base_patch14_224.pt_ee24](https://drive.google.com/file/d/1Ok58RCRvKdq1_VcFn35FQOHyznvq8JFr/view?usp=sharing)      | `gdown 1Ok58RCRvKdq1_VcFn35FQOHyznvq8JFr` |
| Fine-tuned  | CE24         | [eva02_base_patch14_224.ee24_ft_ce24](https://drive.google.com/file/d/123TjuBw-34bKXBu7njzKjbcObNXsnuEY/view?usp=sharing) | `gdown 123TjuBw-34bKXBu7njzKjbcObNXsnuEY` |
